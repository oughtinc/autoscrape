#!/usr/bin/env python3

sample = """<html>
  <div class="article" id="art1">
    <h2 id="yy">John Doe</h2><li>Jane Smith</li>
    <p>Title: <span class="title small">The great article</span><span class="annotation">(out of print)</span></p>
  </div>
  <div class="article">
    <h2><a>Jane Smith</a></h2>
    <p>Title: <span id="x">A small book</span>(out of stock)</p>
  </div>
  <div id="otherel">Something else<a id="fakejane">Jane Smith</a><span>A small book</span></div>
</html>"""

templates = ["""Author: John Doe
Title: The great article""",
"""Author: Jane Smith
Title: A small book"""]

from pprint import pprint, pformat
from lxml import etree
from lxml.etree import XML, ElementTree
import lxml.html
from cssselect import GenericTranslator, SelectorError
import itertools
from functools import lru_cache
import copy
import numpy as np
import sys
import json
import sparse
from profilehooks import profile

def product_of_others(a, axes=None):
    """
    TODO document this
    """
    if axes is None:
        axes = tuple(range(a.ndim))
    if isinstance(axes, int):
        axes = (axes,)

    # flatten the desired axes into one last dimension
    original_shape = a.shape
    other_axes = tuple([ax for ax in range(a.ndim) if ax not in axes])
    new_ax_order = other_axes + axes
    old_ax_order = np.argsort(new_ax_order)
    a = np.transpose(a, new_ax_order)
    a = np.reshape(a, [original_shape[ax] for ax in other_axes] + [np.prod([original_shape[ax] for ax in axes])])

    after = np.concatenate([a[..., 1:], np.ones_like(a[..., 0:1])], axis=-1)
    before = np.concatenate([np.ones_like(a[..., 0:1]), a[..., :-1]], axis=-1)
    after_prod = np.cumprod(after[..., ::-1], axis=-1)[..., ::-1]
    before_prod = np.cumprod(before, axis=-1)

    out = np.reshape(after_prod * before_prod, [original_shape[ax] for ax in other_axes] + [original_shape[ax] for ax in axes])
    out = np.transpose(out, old_ax_order)

    return out

def all_active_combinations(tfl):
    """
    TODO document this
    """
    # all active template combinations
    t_eye = np.eye(tfl.shape[0])[:, :, None, None]
    diag_negator = (1 - 2 * t_eye)

    tfl_diagnegated = (t_eye + diag_negator *
                       np.tile(tfl[:, None, :, :], [1, tfl.shape[0], 1, 1]))

    return tfl_diagnegated

def css(css_selector):
    try:
        return GenericTranslator().css_to_xpath(css_selector)
    except SelectorError:
        print('Invalid selector.')

import re
from collections import defaultdict, namedtuple, OrderedDict


#with open("./tests/websites/fsi.html") as f:
#    sample = f.read()



#templates = ["""Title: Contours and Controversies of Parler
#Authors: David Thiel, Renee DiResta, Shelby Grossman, Elena Cryst""",
#"""Title: DeZurdaTeam: A Twitter network promotes pro-Cuba hashtags (TAKEDOWN)
#Authors: Elena Cryst, Shelby Perkins"""]


def template_prob_match(prob1, prob2):
    total = 0
    for l1, l2 in zip(prob1, prob2):
        total += l1 * l2
    return total


tree = lxml.html.fromstring(sample)

class ContentElement():
    def __init__(self, selector, parent, content, n_fields, n_templates):
        self.parent = parent
        self.selector = selector
        self.content = content
        self.n_fields = n_fields
        self.n_templates = n_templates

        super().__init__()

    def getparent(self):
        return self.parent

    def iterancestors(self):
        return itertools.chain(iter([self.parent]), self.parent.iterancestors())

    def iterchildren(self):
        return iter([])

    def get_xored_template_field_prob(self):
        """
        Return a version of self.template_field_prob that is collapsed along the fields,
        showing the probability that this will take on exactly one field (XOR) for each template.
        """

        def xor_values(probs):
            negated_m = np.tile(np.array(probs)[None, :], [len(probs), 1])
            m = (1 - negated_m) + 2 * np.eye(len(probs)) * (negated_m - .5)
            return np.sum(np.prod(m, 1))

        all_template_prob = [[] for ti in range(self.n_templates)]  # for each template index, a list of all probs any field ever had.
        for field, template_prob in self.template_field_prob.items():
            for ti, prob in enumerate(template_prob):
                all_template_prob[ti].append(prob)

        xored_values = [0] * self.n_templates
        for ti, probs in enumerate(all_template_prob):
            xored_values[ti] = xor_values(probs)

        return xored_values


    def update_template_other_prob(self):
        total = 0
        for field_probs in self.template_field_prob.values():
            for prob in field_probs:
                total += prob

        self.template_other_prob = 1 - total

    def __hash__(self):
        return hash((self.parent, self.selector))

    def __repr__(self):
        return f"<{self.parent.tag} {self.selector}={self.content}>"

class TemplateProb():
    def __init__(self, n_templates):
        self.empty_template_prob = [0] * n_templates
        self.data = {}

    def __getitem__(self, key):
        return self.data[key] if key in self.data else self.empty_template_prob

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return pformat(self.data)


# we first want to find all of the instances of the values.
class PathInductor():
    def __init__(self, tree, templates):
        self.tree = tree
        self.templates = templates
        self.parsed_templates = [self.parse_template(t) for t in templates]
        self.fields = list(sorted(set().union(*[list(tp.keys()) for tp in self.parsed_templates])))
        self.n_fields = len(self.fields)

        self.index_of_field = {f: i for i, f in enumerate(self.fields)}

        self.n_templates = len(self.templates)

        self.n_levels = self.get_max_tree_depth(tree) + 1

        # loop through the elements and index them
        self.element_by_index = list(tree.iter())
        self.index_of_element = {e: i for i, e in enumerate(self.element_by_index)}
        self.n_elements = sum(1 for _ in tree.iter())

        self.field_prob = OrderedDict() # {[element]: np.array(templates, fields, levels)}
        self.children_consider_this_field_parent = {}

        self.item_prob = OrderedDict() # {[element]: np.array(templates, levels)}
        self.children_consider_this_item_parent = {}

        self.item_field_level_prob = defaultdict(float)  # by (field, level) -- this is after checking for presence of children

        self.field_selectors = defaultdict(lambda: defaultdict(float)) # by (field, level)
        self.item_selectors = []

        self.content_elements_by_parent = defaultdict(set)
        self.content_elements_by_content = defaultdict(set)
        self.template_fields_by_value = defaultdict(set)

    def get_max_tree_depth(self, element):
        maxdepth = 1
        for c in element.iterchildren():
            maxdepth = max(maxdepth, 1 + self.get_max_tree_depth(c))
        return maxdepth

    def parse_template(self, template):
        return {l.split(": ", 1)[0]: l.split(": ", 1)[1] for l in template.split("\n")}

    @lru_cache(None)  # elements (e) are hashable!
    def get_element_selector_options(self, e, allow_nonunique_classes=False):
        if isinstance(e, ContentElement):
            return [(None, e.selector)]

        # if there are multiple siblings
        siblings = list(e.itersiblings(e.tag, preceding=False)) + list(e.itersiblings(e.tag, preceding=True))

        selectors = []
        if siblings:
            # filter by same tag
            # does this one have a class that distinguishes it from the others?
            e_classes = set(e.attrib.get("class", "").split())
            some_siblings_classes = set()

            for s in siblings:
                s_classes = s.attrib.get("class", "").split()
                some_siblings_classes = some_siblings_classes.union(*s_classes)

            unique_identifying_classes = frozenset(e_classes.difference(some_siblings_classes))
            if unique_identifying_classes:
                selectors.append(unique_identifying_classes) #{"strategy": "classes", "tag": e.tag, "classes": unique_identifying_classes})

            if allow_nonunique_classes:
                all_siblings_classes = set()
                for s in siblings:
                    s_classes = s.attrib.get("class", "").split()
                    all_siblings_classes = all_siblings_classes.intersection(set(s_classes))
                nonunique_identifying_classes = frozenset(e_classes.difference(unique_identifying_classes).difference(some_siblings_classes))

                if nonunique_identifying_classes:
                    selectors.append(nonunique_identifying_classes)

            parent_tree = etree.ElementTree(e.getparent())
            path_full = parent_tree.getpath(e)
            index_tag = path_full[-len(path_full.split("[")[-1]):-1]
            selectors.append(int(index_tag))

        result = [(e.tag, s) for s in (selectors or [""])]  # when no particular strategies have been identified, just use the string.o
        return result

    def get_xpath_part(self, selector):
        (tag, strategy) = selector
        if tag is None:
            return strategy
        elif isinstance(strategy, set) or isinstance(strategy, frozenset): # classes
            return tag + "[@class and (" + " or ".join([f"contains(concat(' ', normalize-space(@class), ' '), ' {c} ')" for c in list(strategy)]) + ")]"
        elif isinstance(strategy, int): # index
            return tag + "[" + str(strategy) + "]"
        elif strategy == "": # no strategies, just use the tag itself
            return tag

    def compile_xpath(self, selectors, global_path=False):
        return ("//" if global_path else "./") + "/".join(self.get_xpath_part(s) for s in selectors[::-1])

    def find_candidate_content_elements(self, value):
        candidates = []
        for e in self.tree.iter():
            for eti, et in enumerate(e.xpath("text()")):
                if et == value:
                    if eti == 0:
                        le = ContentElement("text()", e, value, self.n_fields, self.n_templates)
                    else:
                        le = ContentElement(f"text()[{eti + 1}]", e, value, self.n_fields, self.n_templates)
                    candidates.append(le)
                    self.content_elements_by_content[value].add(le)
                    self.content_elements_by_parent[e].add(le)

            for eti, et in enumerate(e.xpath("@href")):
                if et == value:
                    le = ContentElement("@href", e, value, self.n_fields, self.n_templates)
                    candidates.append(le)
                    self.content_elements_by_content[value].add(le)
                    self.content_elements_by_parent[e].add(le)

        return candidates

    def initialize_leaf_probs(self):
        for ti, pt in enumerate(self.parsed_templates):
            for field, value in pt.items():
                candidate_content_elements = self.find_candidate_content_elements(value)
                template_prob = 1. / len(candidate_content_elements)
                field_index = self.index_of_field[field]
                for ce in candidate_content_elements:
                    self.field_prob[ce] = np.zeros((self.n_templates, self.n_fields, self.n_levels))
                    self.field_prob[ce][ti, field_index, 0] = template_prob
                    self.template_fields_by_value[value].add((ti, field))

    def update_el_probs(self, element):
        # PARENT CONTRIBUTIONS
        parent = element.getparent()

        parent_item_tl = np.zeros((self.n_templates, self.n_levels))   # zero, unless we know better
        parent_item_tl[:, :-1] = self.item_prob.get(parent, parent_item_tl)[:, 1:]
        parent_item_ancestorness_t = (np.sum(parent_item_tl[:, 1:], (1), keepdims=True)
                                      if parent in self.item_prob else 1)

        parent_field_tfl_raw = np.ones((self.n_templates, self.n_fields, self.n_levels))   # one, unless we know better
        parent_field_tfl_raw[:, :, :-1] = self.field_prob.get(parent, parent_field_tfl)[:, :, 1:]
        parent_no_other_templates_raw = product_of_others(np.prod(1 - parent_field_tfl_raw, (1, 2), keepdims=True))

        parent_field_tfl = (parent_field_tfl_raw * (1 - parent_item_ancestorness_t) +  # initially all one
                            np.ones_like(parent_field_tfl_raw) * parent_item_ancestorness_t)
        parent_field_invtfl = ((1 - parent_field_tfl_raw) * (1 - parent_item_ancestorness_t) +  # initially all one
                            np.ones_like(parent_field_tfl_raw) * parent_item_ancestorness_t)
        parent_no_other_templates = (parent_no_other_templates_raw * (1 - parent_item_ancestorness_t) +  # initially all zero (!?)
                                     np.ones_like(parent_no_other_templates_raw) * parent_item_ancestorness_t)

        # LATERAL (SAME TFL) CONTRIBUTIONS
        no_other_field_competitor = self.field_competitor_tfl_others_off.get(element, 1)
        exactly_one_field_competitor = self.field_competitor_tfl_exactly_one_on.get(element, 1)
        no_other_item_competitor = self.item_competitor_tfl_others_off.get(element, 1)
        exactly_one_other_item_competitor = self.item_competitor_tfl_others_one_on.get(element, 1)

        # CHILDRENS' CONTRIBUTIONS
        children_field_tfl = np.zeros((self.n_templates, self.n_fields, self.n_levels))
        children_item_tfl = np.zeros((self.n_templates, self.n_levels))
        if isinstance(element, lxml.etree.ElementBase):
            for child in itertools.chain(element.iterchildren(), self.content_elements_by_parent[element]):
                child_field_probs, child_item_probs = self.update_el_probs(child)
                if child_field_probs is not None:
                    children_field_tfl[:, :, 1:] += child_field_probs[:, :, :-1]
                if child_item_probs is not None:
                    children_item_tfl[:, 1:] += child_item_probs[:, :-1]
        elif isinstance(element, ContentElement):
            children_field_tfl = self.field_prob[element]  # just the thing itself

        children_item_tfl[:, 0] = np.prod(np.sum(children_field_tfl, (2)), (1))


        # ITEM PROBS -- COMBINING AND NORMALIZING THE CONTRIBUTIONS

        item_tfl_active = children_item_tfl * parent_item_tfl * no_other_item_level * no_other_item_competitor
        item_tfl_inactive = (1 - children_item_tfl) * one_other_item_competitor

        item_tfl_active = np.divide(item_tfl_active, (item_tfl_active +
                                                      item_tfl_inactive), out=item_tfl_active, where=item_tfl_active > 0)


        # FIELD PROBS -- COMBINING AND NORMALIZING THE CONTRIBUTIONS
        _children_field_template_inactive = 1 - np.prod(np.sum(children_field_tfl, (2), keepdims=True), (1), keepdims=True)
        children_no_other_field_templates = product_of_others(_children_field_template_inactive, (0))

        children_no_other_field_levels = product_of_others(children_field_invtfl, 2)  # all the others are off
        _children_only_field_level_active = children_field_tfl * children_no_other_field_levels  # this AND no other level
        children_any_other_or_no_field_level = (np.sum(_children_only_field_level_active, 2, keepdims=True) -
                                                _children_only_field_level_active +  # any other level (and no other but it), or ...
                                                np.prod(children_field_invtfl, 2, keepdims=True)) # ... no level at all

        parent_children_field_templates_agree = (np.prod(_parent_field_template_inactive) * np.prod(_children_field_template_inactive) + # either all inactive, or ...
                                                 np.sum(np.prod(all_active_combinations(_parent_field_template_inactive)  # ... exactly the same is active in both
                                                                * all_active_combinations(_children_field_template_inactive), (0)), (0), keepdims=True))


        field_tfl_active = (parent_field_tfl *
                            parent_no_other_field_templates *
                            children_field_tfl *
                            children_no_other_field_templates *
                            children_no_other_field_levels *
                            no_other_field_competitor)

        field_tfl_first_wrong_child = (parent_field_tfl *  # parent is still correct, but we're the wrong sibling
                                       children_field_invtfl *
                                       exactly_one_field_competitor *  # but not the thing itself. TODO this is initially zero in the first pass, but could be quite a bit higher
                                       parent_no_other_field_templates *
                                       children_no_other_field_templates *
                                       children_no_other_field_levels) # if we've decided that the parent is right, then all the other levels are def. also wrong

        field_tfl_wrong_parent = (parent_field_invtfl *
                                  children_field_invtfl *
                                  exactly_one_field_competitor *  # but not the thing itself
                                  parent_children_field_templates_agree *
                                  children_any_other_or_no_field_level)  # does this doulbe up the problem?

        field_tfl_active = np.divide(field_tfl_active, (field_tfl_active + field_tfl_wrongchild + field_tfl_wrong_parent), out=field_tfl_active, where=field_tfl_active > 0)


        if np.sum(field_tfl_active) or element in self.field_prob:
            # add new ones if they have a nonzero probability of being /something/,
            # or update existing ones (because sometimes we need to update with all-zero).
            # but otherwise, don't waste memory storing all-zero results
            self.field_prob[element] = field_tfl_active

        if np.sum(item_tfl_active) or element in self.item_prob:
            self.item_prob[element] = item_tfl_active

        return (field_tfl_active if np.sum(field_tfl_active) else None, item_tfl_active if np.sum(item_tfl_active) else None)

    def update_field_competitor_probabilities(self, el):
        field_competitor_items = self.field_prob.items()
        field_competitor_tfl = np.zeros((len(field_competitor_items), self.n_templates, self.n_fields, self.n_levels))
        for i, (cel, tfl) in enumerate(field_competitor_items):
            field_competitor_tfl[i, ...] = tfl

        field_competitor_tfl_others_off_full = product_of_others(1 - field_competitor_tfl, (0))  # axis 0 = other elements
        field_competitor_tfl_exactly_one_on_full = np.sum(field_competitor_tfl * field_competitor_tfl_others_off_full, (0), keepdims=True)

        self.field_competitor_tfl_none_on = np.prod(1 - field_competitor_tfl, (0))
        self.field_competitor_tfl_others_off = {}
        self.field_competitor_tfl_others_one_on = {}
        for i, (cel, _) in enumerate(field_competitor_items):
            self.field_competitor_tfl_others_off[cel] = field_competitor_tfl_others_off_full[i, ...]
            self.field_competitor_tfl_exactly_one_on[cel] = field_competitor_tfl_exactly_one_on_full[i, ...]

    def update_item_competitor_probabilities(self, el):
        item_competitor_items = self.item_prob.items()
        item_competitor_tl = np.zeros((len(item_competitor_items), self.n_templates, self.n_levels))
        for i, (cel, tl) in enumerate(item_competitor_items):
            item_competitor_tl[i, ...] = tl

        item_competitor_tl_others_off_full = product_of_others(1 - item_competitor_tl, (0))  # axis 0 = other elements
        item_competitor_tl_others_one_on_full = np.sum(item_competitor_tl * item_competitor_tl_others_off_full, (0), keepdims=True) - (item_competitor_tl * item_competitor_tl_others_off_full)

        self.item_competitor_tl_none_on = np.prod(1 - item_competitor_tl, (0))
        self.item_competitor_tl_others_off = {}
        self.item_competitor_tl_others_one_on = {}
        for i, (cel, _) in enumerate(item_competitor_items):
            self.item_competitor_tl_others_off[cel] = item_competitor_tl_others_off_full[i, ...]
            self.item_competitor_tl_others_one_on[cel] = item_competitor_tl_others_one_on_full[i, ...]

    def update_probs(self, el, initial=False):
        self.update_field_competitor_probabilities(el)
        self.update_item_competitor_probabilities(el)
        self.update_el_probs(tree, consider_parents=False)

    def _print_probs(self, el, depth=0):
        if depth == 0:
            print("Probabilities")
        if el in self.field_prob:
            print(f"{'  ' * depth}<{el.tag}>:")
            print(self.field_prob[el])
        else:
            print(f"{'  ' * depth}<{el.tag}>")

        for c in el.iterchildren():
            self._print_probs(c, depth + 1)

    def update_item_model(self):
        competitor_items = self.field_prob.items()
        competitor_tfl = np.zeros((len(competitor_items), self.n_templates, self.n_fields, self.n_levels))
        for i, (cel, tfl) in enumerate(competitor_items):
            competitor_tfl[i, ...] = tfl

        total_prob_of_field = np.sum(competitor_tfl, (3)) # sum across levels
        total_prob_of_template = np.prod(total_prob_of_field, (2)) # multiply across fields
        prob_of_item = total_prob_of_template * product_of_others(1 - total_prob_of_template, (0))
        any_one_item = np.sum(prob_of_item, (0), keepdims=True)
        prob_of_item = np.divide(prob_of_item, any_one_item, out=prob_of_item, where=prob_of_item > 0)
        print("prob_of_template")
        print(total_prob_of_template)
        print("prob of item is")
        print(prob_of_item)
        print("sum prob of item across elements (for each template)")
        print(np.sum(prob_of_item, 0))



    def generate_field_selectors(self):
        # now we want to update the probabilities of each selector going up
        # for each field and level, we have a dictionary of selectors with their probabilities.
        self.field_selectors = defaultdict(lambda: defaultdict(float)) # by (field, level)
        for field in self.fields:
            elements_to_update_selectors = set()  # set of (element, level)
            for ce, levelprobs in self.field_prob[field].items():
                if not 0 in levelprobs: # we are just looking for elements that occur at level 0, i.e. leaves
                    continue
                elements_to_update_selectors.add((ce, 0))
                for level, ancestor in enumerate(ce.iterancestors(), 1):
                    elements_to_update_selectors.add((ancestor, level))

            for element, level in elements_to_update_selectors:
                options = self.get_element_selector_options(element)
                element_prob = self.field_prob[field][element][level]
                for option in options:
                    self.field_selectors[field, level][option] += element_prob

        # prune selectors with low probabilities
        for (field, level), selector_options in self.field_selectors.items():
            prunable_options = [selector_option for selector_option, prob in selector_options.items() if prob < 0.01]
            for po in prunable_options:
                del selector_options[po]

        # we have a selector for elements that are (field: URL, generation: 2).
        # the correct one is /div[@class='details']/, but because the URL is just a span, there are a lot of
        # grandparents of spans.
        # TODO clean up and prune the list of selectors for each kind of element
        # for each level, we want to look at the probabilities, and only keep the top 10 selectors
        #print("field selectors are:")
        #pprint(dict(self.field_selectors))
        # TODO especially if there are highly common intersections of classes, find the most likely set of classes


    def generate_item_selectors(self):
        # TODO choose items that have enough itemness, pruning some if necessary, and generate selectors for those.
        itemness_by_tag = defaultdict(dict)
        for element, itemness in self.item_prob.items():
            itemness_by_tag[element.tag][element] = itemness

        element_by_tag = defaultdict(list)
        for tag, element_itemness_probs in itemness_by_tag.items():
            # for all of the ones that have the same tag, see which ones have the same parent.
            element_by_parent = defaultdict(list)
            for element, itemness_prob in element_itemness_probs.items():
                element_by_parent[element.getparent()].append((element, itemness_prob,))

            # for each parent, compute the total score:
            parent_scores = defaultdict(float)
            for parent, elementprobs in element_by_parent.items():
                for element, prob in elementprobs:
                    parent_scores[parent] += prob
            # get the parent with the highest score
            best_parent = max(parent_scores, key=parent_scores.get)
            element_by_tag[tag] = element_by_parent[best_parent]

        tag_scores = defaultdict(float)
        for tag, elementprobs in element_by_tag.items():
            for element, prob in elementprobs:
                tag_scores[tag] += prob
        best_tag = max(tag_scores, key=tag_scores.get)
        best_items = [t for (t, p) in element_by_tag[best_tag]]
        # now generate some selector that captures all of the items but nothing else
        best_item_selectors = [self.get_element_selector_options(t) for t in best_items]
        # first, try merging the selectors' class sets
        selector = (best_item_selectors[0][0][0], frozenset(set.intersection(*[set(s[0][1]) for s in best_item_selectors])))
        item_field_level_coverage = defaultdict(set)
        for item in best_items:
            for field, elementprobs in self.field_prob.items():
                for level, prob in elementprobs[item].items():
                    if prob > 0.01:
                        item_field_level_coverage[field].add(level)
        self.item_selectors.append((selector, 1, item_field_level_coverage))


    def select_and_check_item_candidates(self):

        def recursively_detect_coverage(element, field, level):
            if level > 2:
                candidate_children = set()
                for selector in self.field_selectors[field, level - 1].keys():
                    for child in element.xpath(self.compile_xpath([selector])):
                        candidate_children.add(child)
                return any([recursively_detect_coverage(child, field, level - 1) for child in candidate_children])

            elif level == 2:
                # only one needs to match
                leaf_selectors = self.field_selectors[field, 1].keys()  # the text() has been found in p's and in spans. could be both.
                content_element_selectors = self.field_selectors[field, 0].keys()
                for (leaf_selector, content_element_selector) in itertools.product(leaf_selectors, content_element_selectors):
                    if len(element.xpath(self.compile_xpath([content_element_selector, leaf_selector]))):
                        return True
                return False

        # TODO update the itemness probs based on the item selectors
        coverage = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for (selector, prob, item_field_level_coverage) in self.item_selectors:
            for el in tree.xpath(self.compile_xpath([selector], global_path=True)):
                # now, for each one of these found items, we want to check what their itemness would be.
                for field in self.fields:
                    for level in item_field_level_coverage[field]:
                        # we think it might be possible that the item contains (field @ level).
                        # we check this by recursively checking for the presence of such an element.
                        # these items, most likely, are not examples we have already discovered.
                        # so when we discover particular items near the bottom of the chain,
                        # we want to figure out the coverage, which will most likely just be "1",
                        # and we will then normalize this by the end of it.
                        field_level_coverage = recursively_detect_coverage(el, field, level)
                        if field_level_coverage:
                            coverage[field][el][level] = True

        # TODO: don't update coverage for items that we already have a prob for
        # TODO: normalize
        # TODO check if close enough to item model, kick out if not
        # TODO update item model accordingly.

    # now we want to run the loop where we continuously update the probabilities
    def approximate_probabilities(self):
        self.initialize_leaf_probs()  # finds the content elements and computes their probabilities
        self.update_probs(tree, initial=True)
        self._print_probs(tree)
        # now we want to perform belief propagation through the network, until all the probabilities settle.

        n_updates = 0 # self.n_levels
        for i in range(n_updates):
            self.update_probs(tree)
        if n_updates > 0:
            self._print_probs(tree)

        # now let's look at what the items are like.
        self.update_item_model()

        sys.exit()

        # build the initial item model based on full-coverage items only.
        self.generate_field_selectors()

        self.generate_item_selectors()  # find item selectors for the top n items.
        self.select_and_check_item_candidates()  # select a bunch of other candidates for items, and also their item types (i.e. likely level combos)
        return


pi = PathInductor(tree, templates)
pi.approximate_probabilities()

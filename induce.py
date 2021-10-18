#!/usr/bin/env python3

sample = """<html>
  <body>
    <div class="article" id="art1">
      <h2 id="yy">John Doe</h2><li>Jane Smith</li>
      <p>Title: <span class="title small">The great article</span><span class="annotation">(out of print)</span></p>
    </div>
    <div class="article">
      <h2><a>Jane Smith</a></h2>
      <p>Title: <span id="x">A small book</span>(out of stock)</p>
    </div>
    <div id="otherel">Something else<a id="fakejane">Jane Smith</a><span>A small book</span></div>
  <body>
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

from debug_html import render_debug_html, wrap_debug_html

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

def one_of_others_is_on(a, axes=None):
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

    # create an after and a before, and then do a multiplication thing.
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

    def update_el_probs(self, element, print_debug=False):
        # PARENT CONTRIBUTIONS
        parent = element.getparent()

        # probability of this being an item, according to the parent
        item_tl_from_parent = np.ones((self.n_templates, self.n_levels)) / self.n_levels # 17%, because we have no idea, initially
        if parent in self.item_prob:
            item_tl_from_parent[:, :-1] = self.item_prob[parent][:, 1:]    # we then take the item_prob from above and shift it down

        parent_field_tfl_raw = np.ones((self.n_templates, self.n_fields, self.n_levels)) / self.n_levels  # one, unless we know better
        parent_field_tfl_raw[:, :, :-1] = self.field_prob.get(parent, parent_field_tfl_raw)[:, :, 1:]
        parent_no_other_field_templates = product_of_others(np.prod(1 - parent_field_tfl_raw, (1, 2), keepdims=True))

        field_tfl_from_parent = parent_field_tfl_raw
        parent_field_invtfl = (1 - parent_field_tfl_raw)

        # LATERAL (SAME TFL) CONTRIBUTIONS
        no_other_field_competitor = self.field_competitor_tfl_others_off.get(element, 1)
        exactly_one_field_competitor = self.field_competitor_tfl_exactly_one_other_on.get(element, 1) #, np.ones_like(parent_field_tfl))

        no_other_item_competitor = self.item_competitor_tl_others_off.get(element, 1)
        exactly_one_other_item_competitor = self.item_competitor_tl_exactly_one_other_on.get(element, 1) #, np.ones_like(parent_item_tl))

        has_self_field_info = element in self.field_prob
        has_self_item_info = element in self.item_prob

        # CHILDRENS' CONTRIBUTIONS
        children_field_tfls = []
        children_item_tls = []
        children_debug_html = []
        if isinstance(element, lxml.etree.ElementBase):
            for child in itertools.chain(element.iterchildren(), self.content_elements_by_parent[element]):
                child_field_probs, child_item_probs, child_debug_html = self.update_el_probs(child, print_debug)
                if child_field_probs is not None:
                    children_field_tfls.append(child_field_probs)
                if child_item_probs is not None:
                    children_item_tls.append(child_item_probs)
                children_debug_html.append(child_debug_html)

            if len(children_item_tls):
                children_item_tls = np.stack(children_item_tls, axis=-1)
                children_item_tl_exactly_one = np.sum(children_item_tls * product_of_others(children_item_tls, (-1)), -1)
                children_item_tl_none = np.prod(1 - children_item_tls, (-1))
                item_tl_from_children = np.pad(children_item_tl_exactly_one, (0, (1, 0)))[:, :-1]
                no_item_tl_from_children = np.pad(children_item_tl_none, (0, (1, 0)))[:, :-1]
            else:
                item_tl_from_children = np.zeros((self.n_templates, self.n_levels))
                no_item_tl_from_children = np.ones((self.n_templates, self.n_levels))

            if len(children_field_tfls):
                children_field_tfls = np.stack(children_field_tfls, axis=-1)
                # of the children's TFL, exactly one is going to be active (at most).
                children_field_tfl_exactly_one = np.sum(children_field_tfls * product_of_others(children_field_tfls, (-1)), -1)
                children_field_tfl_none = np.prod(1 - children_field_tfls, -1)
                field_tfl_from_children = np.pad(children_field_tfl_exactly_one, (0, 0, (1, 0)))[:, :, :-1]
                no_field_tfl_from_children = np.pad(children_field_tfl_none, (0, 0, (1, 0)))[:, :, :-1]

                # of the children's TFL, we want exactly one of each field to be active; we don't care which one
                children_field_tfl_exactly_one_any_l = np.sum(children_field_tfls * product_of_others(children_field_tfls, (-2, -1)), (-2, -1))
                item_tl_from_children[:, 0] = np.prod(children_field_tfl_exactly_one_any_l, (1))
                no_item_tl_from_children[:, 0] = 1 - item_tl_from_children[:, 0]
            else:
                field_tfl_from_children = np.zeros((self.n_templates, self.n_fields, self.n_levels))
                no_field_tfl_from_children = np.ones((self.n_templates, self.n_fields, self.n_levels))


        elif isinstance(element, ContentElement):
            item_tl_from_children = np.zeros((self.n_templates, self.n_levels))
            no_item_tl_from_children = np.ones((self.n_templates, self.n_levels))
            field_tfl_from_children = self.field_prob[element]  # just the thing itself
            no_field_tfl_from_children = 1 - self.field_prob[element]

        children_debug_html = "".join(children_debug_html)

        no_other_item_level = self.item_competitor_tl_other_levels_off.get(element, 1)

        item_tl_active = (item_tl_from_children *
                          item_tl_from_parent *
                          no_other_item_level *  # we don't want this to be multiple item levels at once.
                          no_other_item_competitor)

        item_tl_inactive = no_item_tl_from_children * (
            exactly_one_other_item_competitor + (1 - item_tl_from_parent) * no_other_item_competitor  # if the parent isn't an item, then it could be that nothing is!
        )

        item_tl_active = np.divide(item_tl_active, (item_tl_active + item_tl_inactive), out=item_tl_active, where=item_tl_active > 0)

        # FIELD PROBS -- COMBINING AND NORMALIZING THE CONTRIBUTIONS
        _children_field_template_inactive = 1 - np.prod(np.sum(field_tfl_from_children, (2), keepdims=True), (1), keepdims=True)
        children_no_other_field_templates = product_of_others(_children_field_template_inactive, (0))
        children_no_other_field_levels = product_of_others(1 - field_tfl_from_children, 2)  # all the others are off

        not_an_item_ancestor = np.prod(1 - item_tl_from_children[:, 1:]) # not an item ancestor for any template or level

        field_tfl_active = (field_tfl_from_parent * #parent_no_other_field_templates) *
                            field_tfl_from_children *
                            not_an_item_ancestor * # if this is an item ancestor (item level >= 1), then it can't be a field ancestor
                            children_no_other_field_templates *
                            children_no_other_field_levels *
                            no_other_field_competitor)

        # how can a thing NOT be the ancestor of a field?
        # 1. its parent says yes, but its children say no. Then it must be somewhere else.
        # 2. its parent says no, and its children say no. Then it must be somewhere or nowhere.

        field_tfl_inactive = no_field_tfl_from_children * (
            exactly_one_field_competitor + (1 - field_tfl_from_parent) * no_other_field_competitor  # if the parent isn't an item, then it could be that nothing is!
        )

        field_tfl_active_unnormalized = np.copy(field_tfl_active)
        field_tfl_active = np.divide(field_tfl_active, (field_tfl_active + field_tfl_inactive), out=field_tfl_active, where=field_tfl_active > 0)

        debug_html = render_debug_html(element.content if isinstance(element, ContentElement) else element.tag,
                                       self.fields,
                                       self.n_templates,
                                       field_tfl_active_unnormalized,
                                       field_tfl_inactive,
                                       field_tfl_active,
                                       field_tfl_from_children,
                                       field_tfl_from_parent,
                                       parent_no_other_field_templates,
                                       not_an_item_ancestor,
                                       children_no_other_field_templates,
                                       no_other_field_competitor if has_self_field_info else -1 * np.ones_like(field_tfl_active),
                                       exactly_one_field_competitor if has_self_field_info else -1 * np.ones_like(field_tfl_active),
                                       children_debug_html,
                                       item_tl_active,
                                       item_tl_from_children,
                                       item_tl_from_parent,
                                       no_other_item_competitor if has_self_item_info else -1 * np.ones_like(item_tl_active),
                                       exactly_one_other_item_competitor if has_self_item_info else -1 * np.ones_like(item_tl_active),
                                       no_other_item_level) if print_debug else ""


        if np.sum(field_tfl_active) or element in self.field_prob:
            # add new ones if they have a nonzero probability of being /something/,
            # or update existing ones (because sometimes we need to update with all-zero).
            # but otherwise, don't waste memory storing all-zero results
            self.field_prob[element] = field_tfl_active

        if np.sum(item_tl_active) or element in self.item_prob:
            self.item_prob[element] = item_tl_active

        return (field_tfl_active if np.sum(field_tfl_active) else None,
                item_tl_active if np.sum(item_tl_active) else None,
                debug_html)


    def item_field_depth_probabilities(self):
        self.item_depth_tfl = np.zeros((self.n_templates, self.n_fields, self.n_levels))

        # what we are interested in is items that have a particular field prob, and that
        # *also* have a particular item prob.
        # We want to multiply the probability of being an item with the probability of being a particular level.
        # What we care about is, what is the probability of being an item-zero of a particular template, and also being of a particular level?
        # So if we look at all of the elements and their probability of being an item-level-zero
        for i, (cel, tl) in enumerate(self.item_prob.items()):
            item_level_zero_probs = tl[:, 0]  # a vector over the templates for this particular element. Maybe 0.4 for T0, 0.1 for T1, 0.0 for T2.
            self.item_depth_tfl += self.field_prob[cel] * item_level_zero_probs[:, None, None]

        self.item_depth_tfl = np.divide(item_depth_tfl, np.sum(item_depth_tfl, (2), keepdims=True), out=item_depth_tfl, where=item_depth_tfl > 0)
        # initially, many of these will be zero divided by zero, and so in that case, the probability needs to have some kind of prior over it.
        # But also, initially, we will only take the children into acocunt anyway. So initially, this doesn't matter at all.
        # the first time around, the whole thing will be zeros, divided by zeros. and so we can just use the divide function, I think?

    def update_field_competitor_probabilities(self):
        field_competitor_items = self.field_prob.items()
        field_competitor_tfl = np.zeros((len(field_competitor_items), self.n_templates, self.n_fields, self.n_levels))
        field_competitor_count = np.zeros((self.n_templates, self.n_fields, self.n_levels))

        for i, (cel, tfl) in enumerate(field_competitor_items):
            field_competitor_tfl[i, ...] = tfl
            field_competitor_count += (tfl > 0)

        field_competitor_tfl_others_off_full = product_of_others(1 - field_competitor_tfl, (0))  # axis 0 = other elements
        only_this_on = field_competitor_tfl * field_competitor_tfl_others_off_full
        # prob that exactly one other one is on, regardless of if this one is on:
        #   first, prob that exactly one *other* one is on.
        #   this is the sum across all elements that only this one is on, not including this particular one.
        field_competitor_tfl_exactly_one_other_on_full = np.sum(only_this_on, (0), keepdims=True) - only_this_on
        # we then divide this by the thing
        # and wherever field_competitor is one, the chance that there is one other, is all wrong.
        # TODO this is potentially all wrong too, because does it just write the result for the right rows and leave everything else ?
        # Need to perhaps write some tests for this
        field_competitor_tfl_exactly_one_other_on_full = np.divide(field_competitor_tfl_exactly_one_other_on_full, (1 - field_competitor_tfl),
                                                                   out=field_competitor_tfl_exactly_one_other_on_full, where=field_competitor_tfl < 1)
        field_competitor_tfl_exactly_one_other_on_full = np.where(field_competitor_tfl == 1, 0, field_competitor_tfl_exactly_one_other_on_full)

        self.field_competitor_tfl_none_on = np.prod(1 - field_competitor_tfl, (0))
        self.field_competitor_tfl_others_off = {}
        self.field_competitor_tfl_exactly_one_other_on = {}
        for i, (cel, _) in enumerate(field_competitor_items):
            self.field_competitor_tfl_others_off[cel] = field_competitor_tfl_others_off_full[i, ...]
            self.field_competitor_tfl_exactly_one_other_on[cel] = field_competitor_tfl_exactly_one_other_on_full[i, ...]

    def update_item_competitor_probabilities(self):
        item_competitor_items = self.item_prob.items()
        item_competitor_tl = np.zeros((len(item_competitor_items), self.n_templates, self.n_levels))
        item_competitor_count = np.zeros((self.n_templates, self.n_levels))
        for i, (cel, tl) in enumerate(item_competitor_items):
            item_competitor_tl[i, ...] = tl
            item_competitor_count += (tl > 0)

        item_competitor_tl_others_off_full = product_of_others(1 - item_competitor_tl, (0))  # axis 0 = other elements
        item_competitor_tl_other_levels_off_full = product_of_others(1 - item_competitor_tl, (2))  # axis 2 = other levels

        only_this_on = item_competitor_tl * item_competitor_tl_others_off_full
        self.item_competitor_tl_none_on = np.prod(1 - item_competitor_tl, (0))

        item_competitor_tl_exactly_one_other_on_full = np.sum(only_this_on, (0), keepdims=True) - only_this_on
        item_competitor_tl_exactly_one_other_on_full = np.divide(item_competitor_tl_exactly_one_other_on_full, (1 - item_competitor_tl),
                                                                   out=item_competitor_tl_exactly_one_other_on_full, where=item_competitor_tl < 1)

        item_competitor_tl_exactly_one_other_on_full = np.where(item_competitor_tl == 1, 0, item_competitor_tl_exactly_one_other_on_full)
        self.item_competitor_tl_others_off = {}
        self.item_competitor_tl_exactly_one_other_on = {}
        self.item_competitor_tl_other_levels_off = {}
        for i, (cel, _) in enumerate(item_competitor_items):
            self.item_competitor_tl_others_off[cel] = item_competitor_tl_others_off_full[i, ...]
            self.item_competitor_tl_exactly_one_other_on[cel] = item_competitor_tl_exactly_one_other_on_full[i, ...]
            self.item_competitor_tl_other_levels_off[cel] = item_competitor_tl_other_levels_off_full[i, ...]

    def update_probs(self, el, print_debug=False):
        self.update_field_competitor_probabilities()
        self.update_item_competitor_probabilities()
        _field_tfl, _item_tfl, debug_html = self.update_el_probs(tree, print_debug)

        if print_debug:
            with open("/tmp/out.html", "w") as f:
                f.write(wrap_debug_html(debug_html))

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
        #self._print_probs(tree)
        # now we want to perform belief propagation through the network, until all the probabilities settle.

        n_updates = 1 # self.n_levels
        for i in range(n_updates):
            self.update_probs(tree, i == n_updates - 1)
        #if n_updates > 0:
        #    self._print_probs(tree)

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

#!/usr/bin/env python3

import pandas as pd

def render_debug_html(title, fields, n_templates,
                      field_tfl_active_unnormalized,
                      field_tfl_inactive,
                      field_tfl_active,
                      children_field_tfl,
                      parent_field_tfl,
                      parent_no_other_field_templates,
                      children_contain_no_item,
                      children_no_other_field_templates,
                      no_other_field_competitor,
                      exactly_one_field_competitor,
                      children_debug_html,
                      item_tl_active,
                      children_item_tl,
                      parent_item_tl,
                      no_other_item_competitor,
                      exactly_one_other_item_competitor,
                      no_other_templates_at_level_zero,
                      no_other_item_level):
    fm = lambda x: "" if x == 0 else '{:.0%}'.format(x)
    return f"""
        <div style="background: rgba(0, 0, 0, 0.04)">
          <h2>{title}</h2>
          <div style="display: flex">
            <div class="datatable">
            <h3>TFL active (pre normalization!)</h3>
            {"".join([pd.DataFrame(field_tfl_active_unnormalized[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>TFL inactive (pre normalization!)</h3>
            {"".join([pd.DataFrame(field_tfl_inactive[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>TFL active (post normalization!)</h3>
            {"".join([pd.DataFrame(field_tfl_active[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
          </div>
          <div style="display: flex; color: navy;">
            <div class="datatable">
            <h3>Children Field TFL</h3>
            {"".join([pd.DataFrame(children_field_tfl[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>Parent Field TFL</h3>
            {"".join([pd.DataFrame(parent_field_tfl[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>parent_no_other_templates</h3>
            {"".join([pd.DataFrame(parent_no_other_field_templates[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>no other field competitor</h3>
            {"".join([pd.DataFrame(no_other_field_competitor[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>exactly one field competitor</h3>
            {"".join([pd.DataFrame(exactly_one_field_competitor[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>children_contain_no_item</h3>
            {"".join([pd.DataFrame(children_contain_no_item[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>children_no_other_field_templates</h3>
            {"".join([pd.DataFrame(children_no_other_field_templates[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
          </div>
          <div style="display: flex; color: darkgreen;">
            <div class="datatable">
                <h3>item_tl_active</h3>
                {pd.DataFrame(item_tl_active).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>children_item_tl</h3>
                {pd.DataFrame(children_item_tl).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>parent_item_tl</h3>
                {pd.DataFrame(parent_item_tl).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>no_other_item_competitor</h3>
                {pd.DataFrame(no_other_item_competitor).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>exactly_one_other_item_competitor</h3>
                {pd.DataFrame(exactly_one_other_item_competitor).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>no_other_templates_at_level_zero</h3>
                {pd.DataFrame(no_other_templates_at_level_zero).to_html(float_format=fm)}
            </div>
            <div class="datatable">
                <h3>no_other_item_level</h3>
                {pd.DataFrame(no_other_item_level).to_html(float_format=fm)}
            </div>
          </div>
          <div style="margin-left: 2rem">
          {children_debug_html}
          </div>
        </div>
        """

def wrap_debug_html(debug_html):
    style = """<style>
    body {
      font-family: Inter;
    }
    .datatable {
      margin: 0 1rem;
    }
    .dataframe {
      font-family: "IBM Plex Mono";
      font-size: 80%;
      border: none;
      border-collapse: collapse;
    }
    .dataframe td {
      border-style: solid;
      width: 2rem;
    }
    </style>"""

    return f"""<html>
            <head><title>Induction output</title>{style}</head>
            <body>{debug_html}</body>
            </html>"""

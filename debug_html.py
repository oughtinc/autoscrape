#!/usr/bin/env python3

import pandas as pd

def render_debug_html(title, fields, n_templates, field_tfl_active,
                      field_tfl_first_wrong_child,
                      field_tfl_wrong_parent,
                      children_field_tfl,
                      parent_field_tfl,
                      parent_no_other_field_templates,
                      children_contain_no_item,
                      children_no_other_field_templates,
                      children_debug_html
                      ):
    fm = lambda x: "" if x == 0 else '{:.0%}'.format(x)
    return f"""
        <div style="background: rgba(0, 0, 0, 0.04)">
          <h2>{title}</h2>
          <div style="display: flex">
            <div class="datatable">
            <h3>TFL active</h3>
            {"".join([pd.DataFrame(field_tfl_active[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>TFL first wrong child</h3>
            {"".join([pd.DataFrame(field_tfl_first_wrong_child[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>TFL wrong parent</h3>
            {"".join([pd.DataFrame(field_tfl_wrong_parent[ti, ...], index=fields).to_html(float_format=fm) for ti in range(n_templates)])}
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
            <h3>children_contain_no_item</h3>
            {"".join([pd.DataFrame(children_contain_no_item[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
            </div>
            <div class="datatable">
            <h3>children_no_other_field_templates</h3>
            {"".join([pd.DataFrame(children_no_other_field_templates[ti, ...]).to_html(float_format=fm) for ti in range(n_templates)])}
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

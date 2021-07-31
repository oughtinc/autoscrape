import requests
from io import StringIO
from datetime import datetime
import time
from lxml import etree
from lxml.etree import XML, ElementTree
import lxml.html
import sys

try:
    import pyppeteer
    import asyncio
    HAS_PYPPETEER = True
except Exception as e:
    print("Could not import Pyppeteer", e)
    HAS_PYPPETEER = False

import jsonpath_ng
import json
import traceback

SEPARATOR = "\n\n\n"

import logging
log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s')

class Extractor:
    def __init__(self):
        self.browser = None
        self.browser_times = []
        self.processing_times = []
        self.headless = True

    async def start_browser(self, headless):
        if self.browser is None or headless != self.headless:
            await self.close_browser()
            log.info(f"Starting headless browser (headless={self.headless})...")
            self.headless = headless
            self.browser = await pyppeteer.launch(headless=self.headless)
            logging.getLogger("pyppeteer").setLevel(logging.WARNING)
            log.info("Browser started.")

    async def close_browser(self):
        if self.browser is not None:
            if HAS_PYPPETEER:
                await self.browser.close()
            self.browser = None
            log.info("\nBrowser closed.")

    async def get_raw_content(self, url, request_strategy="requests", headless=True, delay=0):
        """
        Runs a HTTP(S) request to the url and returns the page's source code.

        For the vast majority of pages, a simple GET request via the requests library is sufficient (request_strategy="requests").

        However, some website require additional interaction, e.g. JS or scrolling,
        either due to overzealous DDOS protections like Cloudflare or due to dynamic loading strategies (Medium blogs are one example).
        For these pages, Selenium can be used.
        """

        start = time.time()
        if request_strategy == "requests":
            r = requests.get(url)
            if not r.status_code == 200:
                logging.warning(f"Could not access {url}. HTTP status {r.status_code}")
                raise LookupError("HTTP non-200 response")
            raw_content = r.text

        elif request_strategy == "chrome" and HAS_PYPPETEER:
            await self.start_browser(headless)
            page = await self.browser.newPage()
            await asyncio.sleep(delay)
            response = await page.goto(url)
            if response.headers["status"] != "200":
                logging.error(f"Status code: {response.headers['status']}")
                logging.error(f"Full headers: {response.headers}")
                raise LookupError
            raw_content = await page.content()
            page = await page.close()

        else:
            logging.warning(f"No viable implementation found for request strategy '{request_strategy}'")
            raw_content = ""
            raise LookupError

        end = time.time()
        self.browser_times.append(end-start)
        return raw_content

    def normalize_space_in_string(self, s):
        try:
            return lxml.html.fromstring(s).xpath("normalize-space(.)") if (len(s) and ord(s[0]) != 65279) else ""
        except lxml.etree.ParserError:
            return ""

    def extract_items_by_xpath(self, item_els, fields):
        item_strs_in_tree = []

        if self.debug and True:
            print("Item:", etree.tostring(item_els[0], pretty_print=True, method="html").decode('unicode_escape'))
        if False:
            print("Whole page:", etree.tostring(tree, pretty_print=True, method="html").decode('unicode_escape'))

        for item_el in item_els:
            item_str = []
            for field in fields:
                if "extractor" in field:
                    result = field["extractor"](item_el.value)
                else:
                    result = item_el.xpath(field['xpath'])
                    if result and len(result):
                        if isinstance(result, list):
                            result = [self.normalize_space_in_string(str(a).strip()) for a in result if a and len(a.strip())]
                            result = ", ".join(result)
                        elif isinstance(result, str):
                            result = self.normalize_space_in_string(str(result).strip())

                if result:
                    item_str.append(f"{field['key']}: {result}")
            item_strs_in_tree.append("\n".join(item_str))

        return item_strs_in_tree

    def extract_items_by_jsonpath(self, item_els, fields):
        item_strs_in_tree = []

        for item_el in item_els:
            item_str = []
            for field in fields:
                if "extractor" in field:
                    result = field["extractor"](item_el.value)
                else:
                    result = jsonpath_ng.parse(field["jsonpath"]).find(item_el.value)
                    if result and len(result):
                        if isinstance(result, list):
                            result = [(r.value or "") for r in result]
                            result = ", ".join(result)

                if result:
                    item_str.append(f"{field['key']}: {result}")
            item_strs_in_tree.append("\n".join(item_str))

        return item_strs_in_tree


    def run(self, spec, debug=False, outfile="/tmp/scraping_results"):
        all_item_strs = []
        self.debug = debug

        logging.info(f"Attempting to scrape {spec['name']} ...")

        if not isinstance(spec["url_funcs"], list):
            spec["url_funcs"] = [spec["url_funcs"]]
        if not isinstance(spec["url_funcs_args"], list):
            spec["url_funcs_args"] = [spec["url_funcs_args"]]

        event_loop = asyncio.get_event_loop()
        try:
            for url_func, url_func_args in zip(spec["url_funcs"], spec["url_funcs_args"]):
                i = -1
                previous_result = None
                next_url_gen = iter(url_func_args) if hasattr(url_func_args, '__iter__') else url_func_args
                while True:
                    i += 1
                    if hasattr(url_func_args, '__iter__'):
                        args = next(next_url_gen, None)
                    else:
                        args = url_func_args(previous_result)

                    if args is None:  # this isn't the same as [None]
                        break

                    if self.debug and i > 1:
                        break

                    url = url_func(*args)

                    if url is None or len(url) == 0:
                        break

                    print(f"[{i}] Items already found: {len(all_item_strs)} (current url: {url})", end="\r", flush=True)

                    try:
                        start = time.time()

                        raw_content_task = event_loop.create_task(self.get_raw_content(url, spec.get("request_strategy", "requests"),
                                                                                                   spec.get("headless", True),
                                                                                                   spec.get("delay", 0)))

                        raw_content = asyncio.run(raw_content_task)

                        if "preprocessor" in spec:
                            raw_content = spec["preprocessor"](raw_content)

                        previous_result = raw_content

                        item_strs = []
                        if spec.get("content_type", "html") == "html":
                            tree = lxml.html.fromstring(raw_content)
                            tree.make_links_absolute(url)
                            item_els = tree.xpath(spec["items_path"])

                            if len(item_els) == 0:
                                logging.info(f"\nNo {'more ' if len(all_item_strs) else ''}articles found, stopping.")
                                break

                            item_strs = self.extract_items_by_xpath(item_els, spec["fields"])

                        elif spec.get("content_type", "json") == "json":
                            item_els = jsonpath_ng.parse(spec["items_path"]).find(json.loads(raw_content))
                            if len(item_els) == 0:
                                logging.info(f"\nNo {'more ' if len(all_item_strs) else ''}articles found, stopping.")
                                break

                            item_strs = self.extract_items_by_jsonpath(item_els, spec["fields"])

                        if item_strs is None:
                            break

                        len_before_new_items = len(all_item_strs)
                        all_item_strs += item_strs

                        # remove duplicates
                        len_before_deduplication = len(all_item_strs)
                        all_item_strs = list(set(all_item_strs))
                        len_after_deduplication = len(all_item_strs)
                        deduplication_difference = len_before_deduplication - len_after_deduplication

                        if len_after_deduplication == len_before_new_items:
                            logging.info("\nNo more new items available, stopping.")
                            raise LookupError
                        elif deduplication_difference > 0:
                            logging.info(f"\nRemoved {deduplication_difference} duplicated item(s)")

                        end = time.time()
                        self.processing_times.append(end - start)

                    except LookupError:
                        break
                    except AttributeError:
                        traceback.print_exc()
                        break
                    except ValueError:
                        traceback.print_exc()
                        break
                    except KeyboardInterrupt:
                        log.info("\nReceived exit, exiting")
                        break
                        event_loop.close()
                        sys.exit()
                    except RuntimeError:
                        break
                        event_loop.close()
                        sys.exit()
                    except Exception as e:
                        log.info(f"\nError while trying to scrape {url}: {e}")
                        traceback.print_exc()
                        continue

            log.info(f"\nProcessing complete. Items found: {len(all_item_strs)}.")
            log.info(f"\nRequest time: {sum(self.browser_times)}. Processing time: {sum(self.processing_times)}")

            if self.debug:
                print(SEPARATOR.join(all_item_strs[0:5]))
            else:
                with open(outfile, "w") as f:
                    f.write(SEPARATOR.join(all_item_strs))

        finally:
            event_loop.run_until_complete(self.close_browser())

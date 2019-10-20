from collections import deque
import json
import os
import re

import wikipediaapi


def remove_parens(str):
    paren = re.compile(r'\([\w|\s]+\)')
    return re.sub(paren, '', str).strip()


class AcronymExpansionScraper:
    """
    This class encapsulates all the functionality related to scraping expansions for medical acronyms from Wikipedia
    It makes use of the python wrapper over Wikimedia APIs - Wikipedia-API
    (https://github.com/martin-majlis/Wikipedia-API/)
    To install the wrapper package run "pip install wikipedia-api"
    """

    def __init__(self):
        """
        The class constructor initializes the wiki library instance
        """
        self.wiki = wikipediaapi.Wikipedia('en')
        self.page = None
        self.medical_term_cues = ['medical', 'medicine', 'biological', 'biology', 'scientific', 'science']
        self.disambiguation_page_suffix = '_(disambiguation)'

    @staticmethod
    def print_sections(sections, level=0):
        """
        A developer utility to print all the sections in a wiki page
        :param sections: List of all sections in a page
        :param level: The nesting level of the first section in sections
        :return: None
        """
        for section in sections:
            print("%s: %s - %s" % ("*" * (level + 1), section.title, section.text[0:40]))
            AcronymExpansionScraper.print_sections(section.sections, level + 1)

    def load_page(self, query):
        """
        This method loads the wiki disambiguation page corresponding to the query and sets the class parameter page
        :param query: The query for which wiki is to be searched
        :return: None
        """
        self.page = self.wiki.page(query + self.disambiguation_page_suffix)
        if not self.page.exists():
            raise Exception(query + ' did not return a valid disambiguation page on Wikipedia')

    def is_medical_section(self, section_title):
        """
        This method analyzes the section's title to determine if it is a medical section
        :param section_title: The title of the section to be analyzed
        :return: Boolean determining if it is a medical section or not
        """
        section_title_lower = section_title.lower()
        cues_found = [True for cue in self.medical_term_cues if cue in section_title_lower]
        if cues_found:
            return True
        return False

    def get_medical_expansions(self):
        """
        This method gets all the medical expansions from the wiki disambiguation page
        :return: A list of all the medical expansions in the page
        """
        if not self.page:
            raise Exception('Wikipedia page uninitialized. Can\'t search page that doesn\'t exist')
        sections_to_scan = deque(self.page.sections)
        while sections_to_scan:
            section = sections_to_scan.pop()
            if self.is_medical_section(section.title):
                if section.sections:
                    for subsection in section.sections:
                        sections_to_scan.appendleft(subsection)
                else:
                    return section.text.split('\n')

    @staticmethod
    def format_wiki_output(sf, expansions):
        expansions = list(map(lambda x: x.lower(), expansions))
        splitters = re.compile(r'/|\bor\b|\band\b')
        formatted_expansions = set()
        for expansion_str in expansions:
            sub_expansions = expansion_str.split(',')[0].strip()
            sub_expansions = re.split(splitters, sub_expansions)
            sub_expansions = list(map(lambda x: x.strip(), sub_expansions))
            for se in sub_expansions:
                if len(se) > 1 and not se == sf.lower():
                    formatted_e = remove_parens(se)
                    formatted_expansions.add(formatted_e)
        return list(formatted_expansions)


def extract_expansions(acronyms, use_cached=True):
    print('Extracting expansions from Wikipedia...')
    out_fn = './data/derived/wikipedia_acronym_expansions.json'
    if use_cached and os.path.exists(out_fn):
        return out_fn
    expansion_scraper = AcronymExpansionScraper()
    acronym_expansions = dict()
    for acronym in acronyms:
        try:
            expansion_scraper.load_page(acronym)
            medical_expansions = expansion_scraper.get_medical_expansions()
            if medical_expansions:
                medical_expansions_formatted = expansion_scraper.format_wiki_output(acronym, medical_expansions)
                acronym_expansions[acronym] = medical_expansions_formatted
        except:
            pass
    with open(out_fn, 'w') as fp:
        json.dump(acronym_expansions, fp)
    return out_fn


if __name__ == '__main__':
    acronyms = json.load(open('../../data/derived/acronyms.json', 'r'))
    extract_expansions(acronyms)

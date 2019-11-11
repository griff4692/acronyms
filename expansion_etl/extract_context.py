import logging
import re
from enum import Enum

import inflect

logger = logging.getLogger(__name__)


class TermType(Enum):
    """
    This is the enumeration for the term type used throughout the project's vocabulary.
    Can be moved to utils later if Griffin is okay with it!
    """
    LONG_FORM = 'lf'
    SHORT_FORM = 'sf'


class ContextType(Enum):
    """
    This is the enumeration for the context type
    """
    WORD = 'word'
    PARAGRAPH = 'paragraph'
    DOCUMENT = 'document'


class ContextExtractor:
    """
    This class is used to extract contexts from a document for a given short-form or long-form of an acronym.
    The main interface to the class will be extract_contexts()!
    """

    def __init__(self):
        self.inflect_engine = inflect.engine()
        self.split_lines_regex = re.compile(r'[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029](?: )*')

    @staticmethod
    def trim_boundaries(string_list):
        """
        This method trims the string list at the head and tail and removes any remaining empty strings or single
        punctuation like period and apostrophe that are the result of string splitting
        :param string_list: A list of strings to trim
        :return: The trimmed list
        """
        if string_list and len(string_list[0]) == 1 and not string_list[0].isalnum():
            string_list = string_list[1:]
        if string_list and not string_list[0]:
            string_list = string_list[1:]
        if string_list and len(string_list[-1]) == 1 and not string_list[-1].isalnum():
            string_list = string_list[:-1]
        if string_list and not string_list[-1]:
            string_list = string_list[:-1]
        return string_list

    def get_context(self, document: str, match_found: re.Match, context_config: dict):
        """
        This method gets the context around a found match in the document in accordance with the context configuration
        :param document: The document to be scanned
        :param match_found: The match that is to be used as the center of the context window
        :param context_config: The context configuration
        :return: A string containing the context around the found match (Can parameterize later to return str or list!)
        """
        preceding_text = document[:match_found.start()]
        succeeding_text = document[match_found.end():]
        if context_config['type'] == ContextType.WORD:
            preceding_text_words = self.trim_boundaries(re.split(r'\W+', preceding_text))
            succeeding_text_words = self.trim_boundaries(re.split(r'\W+', succeeding_text))
            return ' '.join(preceding_text_words[len(preceding_text_words) - context_config['size']:]
                            + succeeding_text_words[:context_config['size']])
        if context_config['type'] == ContextType.PARAGRAPH:
            preceding_text_lines = self.trim_boundaries(re.split(self.split_lines_regex, preceding_text))
            succeeding_text_lines = self.trim_boundaries(re.split(self.split_lines_regex, succeeding_text))
            preceding_text_empty_line_indices = [index for index, item in enumerate(preceding_text_lines)
                                                 if len(item.strip()) < 1]
            succeeding_text_empty_line_indices = [index for index, item in enumerate(succeeding_text_lines)
                                                  if len(item.strip()) < 1]
            if not preceding_text_empty_line_indices:
                preceding_text_empty_line_indices = [-1]
            if not succeeding_text_empty_line_indices:
                succeeding_text_empty_line_indices = [len(succeeding_text_lines)]
            return ' '.join(preceding_text_lines[
                            preceding_text_empty_line_indices[
                                len(preceding_text_empty_line_indices) - context_config['size']
                                if context_config['size'] < len(preceding_text_empty_line_indices) else -1] + 1:] +
                            succeeding_text_lines[:succeeding_text_empty_line_indices[
                                context_config['size'] - 1
                                if context_config['size'] < len(succeeding_text_empty_line_indices) else -1]])

    def get_contexts_for_long_form(self, long_form: str, document: str, context_config: dict, allow_inflections: bool,
                                   ignore_case: bool):
        """
        This method gets the contexts for all occurrences of a long form
        :param long_form: The long form for which contexts are to be returned
        :param document: The document to be scanned
        :param context_config: The context configuration
        :param allow_inflections: Boolean flag to allow inflections on the long form
        :param ignore_case: Boolean flag to ignore the case of the long form
        :return: A string containing the context around the found match (Can parameterize later to return str or list!)
        """
        search_regex = r'(\s|\b)'
        words_in_long_form = long_form.split()
        text_to_search = document.lower() if ignore_case else document
        for word in words_in_long_form:
            if ignore_case:
                base_word_form = word.lower()
            else:
                base_word_form = word
            word_forms = [base_word_form]
            if allow_inflections:
                singular_inflection = self.inflect_engine.singular_noun(base_word_form)
                if singular_inflection:
                    word_forms.append(singular_inflection)
                    word_forms.append(singular_inflection + '\'s')
                plural_inflection = self.inflect_engine.plural_noun(base_word_form)
                word_forms.append(plural_inflection)
                word_forms.append(plural_inflection + '\'')
            search_regex += r'(' + '|'.join(word_forms) + r')(\s+|\b)'
        if context_config['type'] == ContextType.DOCUMENT:
            if re.search(search_regex, text_to_search):
                return [document]
            return []
        else:
            return [self.get_context(document, match, context_config) for match in
                    re.finditer(search_regex, text_to_search)]

    def get_contexts_for_short_form(self, short_form: str, document: str, context_config: dict, allow_inflections: bool,
                                    ignore_case: bool):
        """
        This method gets the contexts for all occurrences of a short form
        :param short_form: The short form for which contexts are to be returned
        :param document: The document to be scanned
        :param context_config: The context configuration
        :param allow_inflections: Boolean flag to allow inflections on the short form
        :param ignore_case: Boolean flag to ignore the case of the short form
        :return: A string containing the context around the found match (Can parameterize later to return str or list!)
        """
        search_regex = r'(\s|\b)'
        text_to_search = document.lower() if ignore_case else document
        if ignore_case:
            base_short_form = short_form.lower()
        else:
            base_short_form = short_form
        search_regex += base_short_form
        if allow_inflections:
            search_regex += r'|' + base_short_form + r's|' + base_short_form + r'\'s|' + base_short_form + r's\''
        search_regex += r'(\s|\b)'
        if context_config['type'] == ContextType.DOCUMENT:
            if re.search(search_regex, text_to_search):
                return [document]
            return []
        else:
            return [self.get_context(document, match, context_config) for match in
                    re.finditer(search_regex, text_to_search)]

    @staticmethod
    def check_config_sanity(context_config: dict):
        """
        This method validates the config parameters and raises an Exception if incorrect values are found
        :param context_config: The context configuration
        :return: None
        """
        if type(context_config['type']) != ContextType:
            raise Exception('Invalid context type. Check class ContextType for accepted values')
        if type(context_config['size']) != int:
            raise Exception('Invalid context size. The size has to be an integer')
        if context_config['size'] == 0:
            raise Exception("Invalid context size. Context size can not be zero")

    def extract_contexts(self, search_term: str, term_type: TermType, document: str, context_config: dict,
                         allow_inflections: bool, ignore_case: bool):
        """
        This method is the main interface method to the context extractor class.
        :param search_term: The term for which contexts are to be extracted
        :param term_type: The TermType of the search term
        :param document: The document to be scanned
        :param context_config: The context configuration
        :param allow_inflections: Boolean flag to allow inflections on the short form
        :param ignore_case: Boolean flag to ignore the case of the short form
        :return: A string containing the context around the found match (Can parameterize later to return str or list!)
        """
        self.check_config_sanity(context_config)
        if term_type == TermType.LONG_FORM:
            return self.get_contexts_for_long_form(search_term, document, context_config, allow_inflections,
                                                   ignore_case)
        if term_type == TermType.SHORT_FORM:
            return self.get_contexts_for_short_form(search_term, document, context_config, allow_inflections,
                                                    ignore_case)
        raise Exception('Invalid term type. Refer to class TermType for accepted values')


if __name__ == '__main__':
    context_extractor = ContextExtractor()
    sample_document = """
    This is the 1st paragraph with a shortform SF.

    This is the 2nd paragraph paragraph with a shortform SFs.

    This is the 3rd paragraph paragraph with a longform.

    This is the 4th paragraph multiple longforms occurrence.
    
    
    """
    sample_document = sample_document.strip()
    sample_context_config = {
        'type': ContextType.WORD,
        'size': 2
    }
    print('[' + '\n'.join(context_extractor.get_contexts_for_short_form('SF', sample_document, sample_context_config,
                                                                        allow_inflections=False,
                                                                        ignore_case=False)) + ']')
    print('[' + '\n'.join((context_extractor.get_contexts_for_long_form('Longform', sample_document,
                                                                        sample_context_config, allow_inflections=False,
                                                                        ignore_case=False))) + ']')
    print('[' + '\n'.join((context_extractor.get_contexts_for_long_form('Longform', sample_document,
                                                                        sample_context_config, allow_inflections=False,
                                                                        ignore_case=True))) + ']')
    print('[' + '\n'.join((context_extractor.get_contexts_for_long_form('Longform', sample_document,
                                                                        sample_context_config, allow_inflections=True,
                                                                        ignore_case=False))) + ']')
    print('[' + '\n'.join((context_extractor.get_contexts_for_long_form('Longform', sample_document,
                                                                        sample_context_config, allow_inflections=True,
                                                                        ignore_case=True))) + ']')

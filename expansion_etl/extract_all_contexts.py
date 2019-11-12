from extract_context import ContextExtractor, ContextType

BOUNDARY = '<DOCSTART>'


def get_contexts(expansion, src_file, context_config, allow_inflections=False, ignore_case=True):
    context_extractor = ContextExtractor()
    contexts = []
    with open(src_file, 'r') as f:
        document = ""
        for line in f:
            if line.startswith(BOUNDARY):
                context = context_extractor.get_contexts_for_long_form(expansion, document,
                                                             context_config, allow_inflections=allow_inflections,
                                                             ignore_case=ignore_case)
                contexts.append(context)  # What if > 1 context per document?
                document = ""
            else:
                document += line
        # Context from last document not extracted
    context = context_extractor.get_contexts_for_long_form(expansion, document,
                                                           context_config, allow_inflections=allow_inflections,
                                                           ignore_case=ignore_case)
    contexts.append(context)
    return contexts


def get_context_config(context_type, size):
    context_config = {
        'type': context_type,
        'size': size
    }
    return context_config


if __name__ == "__main__":
    SOURCE_FILE = './data/clean_text.txt'
    sample_context_config = get_context_config(ContextType.WORD, 2)
    print(get_contexts("Admission Date", SOURCE_FILE, sample_context_config))
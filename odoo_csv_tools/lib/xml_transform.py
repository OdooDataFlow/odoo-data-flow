#-*- coding: utf-8 -*-
from . import transform
from collections import OrderedDict
from lxml import etree


class XMLProcessor(transform.Processor):
    def __init__(self, filename, root_node_path, conf_file=False):  # Add conf_file parameter
        super().__init__(filename=filename)  # Call Processor's __init__
        self.root = etree.parse(filename)
        self.root_path = root_node_path
        self.file_to_write = OrderedDict()
        self.conf_file = conf_file  # Initialize conf_file

    def process(self, mapping, filename_out, import_args, t='list', null_values=['NULL', False], verbose=True, m2m=False):
        """
        Transforms data from the XML file based on the provided mapping.

        Args:
            mapping (dict): A dictionary that defines how data from the XML file
                          should be mapped to fields in the output format (e.g., CSV).
                          The keys of the dictionary are the target field names,
                          and the values are XPath expressions to extract the
                          corresponding data from the XML.
            filename_out (str): The name of the output file where the transformed
                              data will be written.
            import_args (dict): A dictionary containing arguments that will be
                               passed to the `odoo_import_thread.py` script
                               (e.g., `{'model': 'res.partner', 'context': "{'tracking_disable': True}"}`).
            t (str, optional): This argument is kept for compatibility but is not
                              used in `XMLProcessor`. Defaults to 'list'.
            null_values (list, optional): This argument is kept for compatibility
                                        but is not used in `XMLProcessor`.
                                        Defaults to `['NULL', False]`.
            verbose (bool, optional): This argument is kept for compatibility but
                                    is not used in `XMLProcessor`. Defaults to
                                    `True`.
            m2m (bool, optional): This argument is kept for compatibility but is
                                 not used in `XMLProcessor`. Defaults to `False`.

        Returns:
            tuple: A tuple containing the header (list of field names) and the
                   transformed data (list of lists).

        Important Notes:
            - The `t`, `null_values`, `verbose`, and `m2m` arguments are present
              for compatibility with the `Processor` class but are not actually
              used by the `XMLProcessor`.
            - The `mapping` dictionary values should be XPath expressions that
              select the desired data from the XML nodes.
        """
        header = mapping.keys()
        lines = []
        for r in self.root.xpath(self.root_path):
            line = [r.xpath(mapping[k])[0] for k in header]
            lines.append(line)
        self._add_data(header, lines, filename_out, import_args)
        return header, lines

    def split(self, split_fun):
        raise NotImplementedError("Method split not supported for XMLProcessor")

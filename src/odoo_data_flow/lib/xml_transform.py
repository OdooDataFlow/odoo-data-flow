"XML Transformations."

from collections import OrderedDict
from typing import Any, Union

from lxml import etree

from . import transform


class XMLProcessor(transform.Processor):
    """Process XML files."""

    def __init__(self, filename, root_node_path, conf_file=False):
        super().__init__(filename=filename)
        parser = etree.XMLParser(
            resolve_entities=False,  # Already had this
            no_network=True,  # Prevent external network access
            dtd_validation=False,  # Disable DTD validation
            load_dtd=False,  # Do not load external DTDs
            # Optionally, you might want to limit recursion for DoS protection
            # huge_tree=False,    # Prevents parsing excessively large documents
            # max_depth=X,        # Max depth of XML tree
            # max_element=Y       # Max number of elements
        )
        self.root = etree.parse(filename, parser=parser)  # noqa nosec S320
        self.root_path = root_node_path
        self.file_to_write = OrderedDict()
        self.conf_file = conf_file

        pass

    def process(
        self,
        mapping: dict[str, str],  # XPath expressions are strings
        filename_out: str,
        import_args: dict[
            str, Any
        ],  # Arguments passed to another script, can be anything
        t: str = "list",
        null_values: Union[
            list[Any], None
        ] = None,  # Or List[str] if specific types are expected
        verbose: bool = True,
        m2m: bool = False,
    ) -> tuple[
        list[str], list[list[str]]
    ]:  # Returns header (list of strings) and lines (list of lists of strings)
        """Transforms data from the XML file based on the provided mapping.

        Args:
            mapping: A dictionary that defines how data from the XML file
                     should be mapped to fields in the output format
                     (e.g., CSV). The keys are target field names,
                     values are XPath expressions.
            filename_out: The name of the output file.
            import_args: Arguments passed to the `odoo_import_thread.py` script.
            t: This argument is kept for compatibility but is not used in
            `XMLProcessor`.
            null_values: This argument is kept for compatibility but is not used
            in `XMLProcessor`.
            verbose: This argument is kept for compatibility but is not used
            in `XMLProcessor`.
            m2m: This argument is kept for compatibility but is not used in
            `XMLProcessor`.

        Returns:
            A tuple containing the header (list of field names) and the
            transformed data (list of lists).

        Important Notes:
            - The `t`, `null_values`, `verbose`, and `m2m` arguments are present
              for compatibility with the `Processor` class but are not actually
              used by the `XMLProcessor`.
            - The `mapping` dictionary values should be XPath expressions that
              select the desired data from the XML nodes.
        """
        if null_values is None:
            null_values = ["NULL", False]
        header = list(mapping.keys())  # mapping.keys() returns a dict_keys object,
        # convert to list if needed
        lines = []
        for r in self.root.xpath(self.root_path):
            # Ensure the XPath expression returns a list, even if it's a single
            # element and handle cases where it might return nothing.
            extracted_values = []
            for k in header:
                # XPath expressions often return a list. Get the first element
                # if found.
                # Or handle multiple elements based on your data structure.
                result = r.xpath(mapping[k])
                extracted_values.append(
                    result[0] if result else ""
                )  # Default to empty string if not found
            lines.append(extracted_values)

        self._add_data(header, lines, filename_out, import_args)
        return header, lines

    def split(self, split_fun):
        """Method split not supported for XMLProcessor."""
        raise NotImplementedError("Method split not supported for XMLProcessor")

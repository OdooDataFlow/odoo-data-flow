```{include} ../README.md
---
end-before: <!-- github-only -->
---
```

# Odoo Data Flow

**A robust, declarative library for managing complex data imports and exports with Odoo.**

Odoo Data Flow is a powerful and flexible Python library designed to simplify the import and export of data to and from Odoo. It allows you to define data mappings and transformations in a declarative way, making complex data operations manageable and repeatable.
You can easily manage complex transformations, relationships, and validations, making your data integration tasks simpler and more reliable.

This library is the successor to the `odoo-csv-import-export` library, refactored for modern development practices and enhanced clarity.

```{mermaid}
---
config:
  theme: redux
---
flowchart TD
 subgraph subGraph0["External Data"]
        A["CSV / XML File"]
  end
 subgraph s1["odoo-data-flow"]
        B{"Model Definition in Python"}
        C["@field Decorators"]
        D["Transformation & Validation Logic"]
  end
 subgraph Odoo["Odoo"]
        E["Odoo Database"]
  end
    A --> B
    B -- Defines --> C
    C -- Applies --> D
    B -- Orchestrates --> E
    A@{ shape: doc}
    E@{ shape: cyl}
    style A fill:#FFF9C4
    style B fill:#C8E6C9
    style E fill:#AA00FF
    style s1 fill:#BBDEFB
    style Odoo fill:transparent
    style subGraph0 fill:transparent


```


## Getting Started

Ready to simplify your Odoo data integrations?

| Step                                       | Description                                                     |
| ------------------------------------------ | --------------------------------------------------------------- |
| 🚀 **[Quickstart](./quickstart.md)**       | Your first end-to-end example. Go from file to Odoo in minutes. |
| ⚙️ **[Installation](./installation.md)**   | How to install the library in your project.                     |
| 🧠 **[Core Concepts](./core_concepts.md)** | Understand the key ideas behind the library.                    |

[license]: license
[contributor guide]: contributing
[command-line reference]: usage

```{toctree}
---
hidden:
maxdepth: 1
---

installation
quickstart
usage
core_concepts
comparison_with_other_tools
guides/index
reference
faq
contributing
Code of Conduct <codeofconduct>
License <license>
Changelog <https://github.com/OdooDataFlow/odoo-data-flow/releases>
ROADMAP
```

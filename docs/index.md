```{include} ../README.md
---
end-before: <!-- github-only -->
---
```

# Odoo Data Flow

**A robust Python toolkit for high-performance, intelligent Odoo data workflows.**

Odoo Data Flow is a powerful and flexible library designed to simplify complex data imports and exports with Odoo. It features a smart import engine with automatic error recovery, multi-threading, and a two-pass strategy for relational data, allowing you to manage complex transformations and validations with confidence.

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

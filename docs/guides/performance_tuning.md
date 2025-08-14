# Guide: Performance Tuning

When working with large datasets, the performance of your data import can become critical. This guide covers the key parameters and strategies you can use to tune the import process for maximum speed and efficiency.

The primary way to control performance is by adjusting the parameters passed to the `odoo-data-flow import` command, which you can set in the `params` dictionary in your `transform.py` script.

---

## Using Multiple Workers

The most significant performance gain comes from parallel processing. The import client can run multiple "worker" processes simultaneously, each handling a chunk of the data.

- **CLI Option**: `--worker`
- **`params` Key**: `'worker'`
- **Default**: `1`

By increasing the number of workers, you can leverage multiple CPU cores on the machine running the import script and on the Odoo server itself.

### Example

To use 4 parallel processes for an import:

```python
# In your transform.py script

import_params = {
    'model': 'sale.order',
    'worker': 4, # Use 4 workers
    # ... other params
}

processor.process(
    mapping=my_mapping,
    filename_out='data/sale_order.csv',
    params=import_params
)
```

This will add the `--worker=4` flag to the command in your generated `load.sh` script.

### Trade-offs and Considerations

- **CPU Cores**: A good rule of thumb is to set the number of workers to be equal to, or slightly less than, the number of available CPU cores on your Odoo server.
- **Database Deadlocks**: The biggest risk with multiple workers is the potential for database deadlocks. This can happen if two workers try to write records that depend on each other at the same time. The library's two-pass error handling system is designed to mitigate this.

## Solving Concurrent Updates with `--groupby`

The `--groupby` option is a powerful feature designed to solve the "race condition" problem that occurs during high-performance, multi-worker imports.

- **CLI Option**: `--groupby`
- **`params` Key**: `'groupby'` (Note: use `groupby`, not `split`)
- **Default**: `None`

### The Problem: A Race Condition

Imagine you are using multiple workers to import contacts that all link to the _same_ parent company.

- **Worker 1** takes a contact and tries to update "Company A".
- At the exact same time, **Worker 2** takes another contact and _also_ tries to update "Company A".

The database locks the company record for Worker 1, so when Worker 2 tries to access it, it fails with a "concurrent update" error.

#### The Solution: The "Sorting Hat"

The `--groupby` option acts like a "sorting hat." Before the import begins, it looks at the column you specify (e.g., `parent_id/id`) and ensures that **all records with the same value in that column are sent to the exact same worker.**

This guarantees that two different workers will never try to update the same parent record at the same time, completely eliminating these errors.

#### Visualizing the Difference

```{mermaid}
---
config:
  theme: redux
---
graph TD
    subgraph subGraph0["Without --groupby (High Risk of Error)"]
        A["Records:<br>C1 (Parent A)<br>C2 (Parent B)<br>C3 (Parent A)"] --> B{Random Distribution};
        B --> W1["Worker 1 gets C1"];
        B --> W2["Worker 2 gets C3"];
        B --> W3["Worker 3 gets C2"];
        W1 -- "tries to update" --> P_A(("Parent A"));
        W2 -- "tries to update" --> P_A;
        W3 -- "updates" --> P_B(("Parent B"));
        P_A --> X["<font color=red><b>ERROR</b></font><br>Concurrent Update"];
    end

    subgraph subGraph1["With --groupby=parent_id/id (Safe)"]
        C["Records:<br>C1 (Parent A)<br>C2 (Parent B)<br>C3 (Parent A)"] --> D{Smart Distribution};
        D -- "parent_id = A" --> W3b["Worker 1 gets C1, C3"];
        D -- "parent_id = B" --> W4b["Worker 2 gets C2"];
        W3b --> S1[("Update Parent A")];
        W4b --> S2[("Update Parent B")];
        S1 & S2 --> Y(["<font color=green><b>SUCCESS</b></font>"]);
    end
    style W1 fill:#FFF9C4
    style W2 fill:#C8E6C9
    style W3 fill:#FFE0B2
    style W3b fill:#FFF9C4
    style W4b fill:#C8E6C9
    style D fill:#BBDEFB
    style B fill:#BBDEFB
    style subGraph0 fill:transparent
    style subGraph1 fill:transparent
    style Y stroke:#00C853
```

### Example

To safely import contacts in parallel, grouped by their parent company:

```python
# In your transform.py script

import_params = {
    'model': 'res.partner',
    'worker': 4,
    # This is the crucial part
    'groupby': 'parent_id/id', # The internal key is 'groupby'
}
```

This will add `--groupby=parent_id/id` to your generated `load.sh` script.

## Understanding Batch Size (`--size`)

The `--size` option is one of the most critical parameters for controlling the performance and reliability of your imports. In simple terms, it controls **how many records are processed in a single database transaction**.

To understand why this is so important, think of it like going through a checkout at a grocery store.

- **CLI Option**: `--size`
- **`params` Key**: `'size'`
- **Default**: `1000` (or the default set in the application's configuration)

### The Default Odoo Behavior: One Big Basket

When you use Odoo's standard import wizard, it's like putting all of your items (every single row in your file) into **one giant shopping basket**. This "all-or-nothing" approach has two major problems:

1.  **Transaction Timeouts:** The Odoo server has a time limit to process your entire basket. If you have too many items (a very large file), it might take too long, and the server will give up with a "Transaction timed out" error. None of your records are imported.
2.  **Single Point of Failure:** If just one record in your giant basket is "bad" (e.g., a missing price), the server rejects the **entire basket**. All of your other perfectly good records are rejected along with the single bad one.

#### How `--size` Solves the Problem: Multiple Small Baskets

The `odoo-data-flow` library allows you to break up your import into smaller, more manageable chunks. When you use `--size 100`, you are telling the tool to use **multiple, smaller baskets**, each containing only 100 items.

This solves both problems:

1.  Each small basket is processed very quickly, avoiding server timeouts.
2.  If one small basket has a bad record, only that basket of 100 records is rejected. All the other baskets are still successfully imported.

#### Visualizing the Difference

```{mermaid}
---
config:
  theme: redux
---
flowchart TD
  subgraph subGraph0["Default Odoo Import (One Big Basket)"]
          B{"One Large Transaction<br>Size=1000"}
          A["1000 Records"]
          D@{ label: "<font color="red"><b>FAIL</b></font><br>All 1000 records rejected" }
          C["Odoo Database"]
    end
  subgraph subGraph1["odoo-data-flow with --size=100 (Multiple Small Baskets)"]
          F{"Transaction 1<br>100 records"}
          E["1000 Records"]
          G["Odoo Database"]
          H{"Transaction 2<br>100 records"}
          I@{ label: "<font color="red"><b>FAIL</b></font><br>Only 100 records rejected" }
          J["...continues with Transaction 3"]
    end
      A --> B
      B -- Single Error --> D
      B -- No Errors --> C
      E --> F
      F --> G & H
      H -- Single Error --> I
      H -- No Errors --> G
      I --> J
      J --> G

      D@{ shape: rect}
      C@{ shape: cyl}
      G@{ shape: cyl}
      I@{ shape: rect}
      style C fill:#AA00FF
      style G fill:#AA00FF
      style subGraph0 fill:transparent
      style subGraph1 fill:transparent

```

#### Trade-offs and Considerations

- **Larger Batch Size**: Can be faster as it reduces the overhead of creating database transactions, but consumes more memory. If one record in a large batch fails, Odoo may reject the entire batch.
- **Smaller Batch Size**: More resilient to individual record errors and consumes less memory, but can be slower due to increased network overhead.
- **WAN Performance:** For slow networks, sending smaller chunks of data is often more stable than sending one massive payload.


### Handling Server Timeouts (`limit-time-real`)

A common source of import failures, especially with large or complex data, is the Odoo server's built-in request timeout.

- **What it is**: Odoo servers have a configuration parameter called `limit-time-real` which defines the maximum time (in seconds) a worker process is allowed to run before it is automatically terminated. The default value is **120 seconds (2 minutes)**.

- **The Problem**: If a single batch of records takes longer than this limit to process (due to complex computations, custom logic, or a very large batch size), the server will kill the process, and your import will fail for that batch.

- **The Solution**: The solution is to reduce the batch size using the `--size` option. By sending fewer records in each transaction, you ensure that each individual transaction can be completed well within the server's time limit.

> **Tip:** If your imports are failing with "timeout" or "connection closed" errors, the first thing you should try is reducing the `--size` value (e.g., from `1000` down to `200` or `100`).


## Mapper Performance

The choice of mappers can impact performance.

- **Fast Mappers**: Most mappers, like `val`, `const`, `concat`, and `num`, are extremely fast as they operate only on the data in the current row.

- **Slow Mappers**: The `mapper.relation` function should be used with caution. For **every single row**, it performs a live search request to the Odoo database, which can be very slow for large datasets.

**Recommendation**: If you need to map values based on data in Odoo, it is much more performant to first export the necessary mapping data from Odoo (e.g., using `odoo-data-flow export`) into a Python dictionary or a separate CSV file, and then use the much faster `mapper.map_val` or other in-memory lookups to do the translation.

---
## Performance Strategy for Relational Data (Automatic Two-Pass Import)

A common performance trap when importing data is writing to relational fields (like `parent_id`) that have an inverse relation (like `child_ids`). In a single pass, updating the `parent_id` for 500 child records could cause Odoo to re-write the `child_ids` list on the single parent record 500 times, slowing the import to a crawl.

`odoo-data-flow` solves this problem **automatically** with its smart, two-pass import engine.

### The New Workflow: Automatic and Efficient

When the pre-flight check detects a self-referential or `many2many` field, the importer automatically switches to this high-performance, two-pass strategy:

1.  **Pass 1 (Create):** The tool **automatically excludes the relational fields** from the initial import. It then uses the fast, multi-threaded `load` method to create all the base records. This completely avoids the slow, cascading update problem.

2.  **Pass 2 (Write):** After all records have been created, the tool performs a second, multi-threaded `write` pass that efficiently sets the relational fields (e.g., `parent_id`) on all the newly created records.

Because this process is automatic, you no longer need to manually use `--ignore` as a performance workaround for these types of fields.

### The Correct Use of `--ignore`

With the new smart importer, the `--ignore` option should be used for its original purpose: to **completely exclude a column from the import process**. Use it for source columns that you do not want to be sent to Odoo in either Pass 1 or Pass 2.


- **CLI Option**: `--ignore`
- **`params` Key**: `'ignore'`

```python
# In your transform.py script

# The mapping still defines the relationship
my_mapping = {
    'id': mapper.m2o_map('child_', 'Ref'),
    'name': mapper.val('Name'),
    'parent_id/id': mapper.m2o_map('parent_', 'ParentRef'), # Define the mapping
}

# The params tell the client to IGNORE the parent_id/id field during import
import_params = {
    'model': 'res.partner',
    'ignore': 'parent_id/id', # The field to ignore for direct import
}

processor.process(
    mapping=my_mapping,
    filename_out='data/contacts.csv',
    params=import_params
)
```

This will generate a `load.sh` script with the `--ignore=parent_id/id` flag. The import client will then skip this column, avoiding the cascading updates entirely.

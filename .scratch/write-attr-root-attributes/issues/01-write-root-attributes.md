# 01: Root attributes are writable, and only root writes create the file

**What to build:** `write_attr(path, "/", name, value)` sets an attribute on the file
root instead of failing. Today the module validates its `var_name` by looking it up in
the file's cached metadata, and a file root is not a variable, so the root path always
raises `KeyError` — even though the README advertises it as a supported call. This
ticket makes the root path work, and settles which writes are allowed to create a file:
a root write to a path that does not exist creates the file with the attribute in it,
while a variable write to a path that does not exist still fails with `FileNotFoundError`
and leaves nothing behind.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `write_attr(path, "/", name, value)` sets a root attribute that reads back through
      the public attribute readers — on a file that has variables, and on a file that has
      none (the purest repro, since nothing could ever match a variable named `/`).
- [ ] Pre-existing root attributes survive a root write.
- [ ] A root write to a path that does not exist creates the file; the new file has no
      variables, and the root attribute reads back.
- [ ] A second root write to that newly created file keeps both attributes, exercising
      re-entry through the metadata cache.
- [ ] A variable write to a path that does not exist raises `FileNotFoundError` and does
      not create the file.
- [ ] The existing test that accepts either `KeyError` or `FileNotFoundError` for an
      unknown variable in an existing file is tightened to `KeyError`, with a match on the
      variable name.
- [ ] A write-then-read-back against cached metadata returns the new value, so the cache
      is not serving stale attributes.
- [ ] `write_attr`'s docstring states which writes may create a file.
- [ ] The shared empty-file test helper is hoisted out of the save-tests section so both
      the root-write and save tests can use it.
- [ ] The "Known Issue" entry in AGENTS.md, which claims a syntax error at a specific
      line of core.py, is corrected or removed — that line does not exist.

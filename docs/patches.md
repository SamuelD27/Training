# Patches Documentation

This file documents any patches or modifications made to upstream training code.

> **Policy**: Prefer config/flags over code modifications. Only patch when absolutely necessary.

---

## Current Patches

**None** - All functionality achieved through configuration and CLI flags.

---

## Patch Policy

### When to Patch

Only patch upstream code when:
1. A critical bug prevents training
2. No config/flag workaround exists
3. The patch is minimal and clearly justified

### When NOT to Patch

Do not patch for:
1. Convenience or preference
2. Features that can be achieved via config
3. Experimental modifications
4. "Improvements" to upstream logic

### Patch Requirements

If a patch is necessary:

1. **Document here** with:
   - File patched
   - Reason for patch
   - Original code
   - Modified code
   - Date and author

2. **Keep it minimal**:
   - Change as few lines as possible
   - Don't refactor or "improve" surrounding code

3. **Make it reversible**:
   - Store original file as `.orig`
   - Provide unpatch instructions

4. **Test thoroughly**:
   - Verify training still works
   - Verify patch solves the issue

---

## Patch Template

```markdown
### Patch: [Brief Description]

**Date**: YYYY-MM-DD
**Author**: [Name]
**Upstream Version**: [commit hash]

**File**: `third_party/sd-scripts/path/to/file.py`

**Reason**:
[Explain why this patch is necessary and why config/flags don't work]

**Original Code**:
```python
# Lines XX-YY
original code here
```

**Modified Code**:
```python
# Lines XX-YY
modified code here
```

**Verification**:
- [ ] Training completes without error
- [ ] Issue is resolved
- [ ] No regression in quality

**Removal Conditions**:
[When can this patch be removed? e.g., "When upstream fixes issue #123"]
```

---

## Known Issues (Not Patched)

Issues we're aware of but have workarounds for:

### Issue: [Example - LR scheduler not applying correctly]
**Workaround**: Use explicit `--lr_scheduler_num_cycles` flag
**Upstream Issue**: N/A
**Status**: Workaround sufficient

---

## Upstream Tracking

We track these upstream issues that may require patches in the future:

| Issue | Description | Status | Our Workaround |
|-------|-------------|--------|----------------|
| - | - | - | - |

---

## Patch Application

If patches exist, they are applied automatically during Docker build.

To apply manually:
```bash
cd /workspace/sd-scripts
patch -p1 < /workspace/lora_training/patches/patch_name.patch
```

To revert:
```bash
cd /workspace/sd-scripts
patch -R -p1 < /workspace/lora_training/patches/patch_name.patch
```

---

## Version Compatibility

Patches are tested against:
- **sd-scripts version**: v0.9.2
- **Python**: 3.10
- **PyTorch**: 2.5.1

If upgrading sd-scripts, re-verify all patches apply cleanly.

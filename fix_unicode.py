#!/usr/bin/env python3
"""
Script to fix Unicode encoding issues in reporting.py
Replaces all pdf.cell() and pdf.multi_cell() with safe versions
"""

import re

# Read the file
with open('backend/pipeline/reporting.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Count occurrences before
cell_count = len(re.findall(r'pdf\.cell\(', content))
multi_cell_count = len(re.findall(r'pdf\.multi_cell\(', content))

print(f"Found {cell_count} pdf.cell() calls")
print(f"Found {multi_cell_count} pdf.multi_cell() calls")

# Replace pdf.cell( with pdf.safe_cell(
# But NOT pdf.safe_cell( (already safe)
content = re.sub(r'(?<!safe_)pdf\.cell\(', 'pdf.safe_cell(', content)

# Replace pdf.multi_cell( with pdf.safe_multi_cell(
# But NOT pdf.safe_multi_cell( (already safe)
content = re.sub(r'(?<!safe_)pdf\.multi_cell\(', 'pdf.safe_multi_cell(', content)

# Write back
with open('backend/pipeline/reporting.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Count after
cell_count_after = len(re.findall(r'(?<!safe_)pdf\.cell\(', content))
multi_cell_count_after = len(re.findall(r'(?<!safe_)pdf\.multi_cell\(', content))

print(f"\n✅ Replaced {cell_count - cell_count_after} pdf.cell() calls")
print(f"✅ Replaced {multi_cell_count - multi_cell_count_after} pdf.multi_cell() calls")
print(f"\nRemaining unsafe calls:")
print(f"  - pdf.cell(): {cell_count_after}")
print(f"  - pdf.multi_cell(): {multi_cell_count_after}")
print("\n🎉 Done! Restart your backend server.")

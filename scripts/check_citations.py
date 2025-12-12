#!/usr/bin/env python3
"""Check which bibliography entries are cited vs orphaned."""

# All bibliography entry keys from references.bib
bib_entries = [
    'Ainslie2023', 'Belinkov2022', 'Beltagy2020', 'Cover2006', 'Dai2019',
    'Dao2023', 'Medusa2024', 'cai2024medusa', 'EAGLE2024', 'PSLM2024',
    'Goyal2021', 'Vapnik2015', 'LopezPaz2015', 'Deng2025', 'Geifman2017',
    'Jin2025', 'Kang2025', 'Kwon2023', 'Liu2025a', 'Liu2025b', 'Ning2023',
    'ning2023skeleton', 'Ren2023', 'Stern2018', 'Yan2024', 'Yoshikawa2023',
    'survey2025', 'Li2024lookahead', 'Rodionov2025', 'Xiao2025sprint',
    'Zheng2025', 'Wei2025sidechannel', 'Chen2018', 'Miyato2018', 'Fazlyab2019',
    'snc_code', 'leviathan2023fast', 'hu2021lora', 'houlsby2019parameter',
    'bulatov2022recurrent', 'gu2023mamba', 'bachlechner2021rezero'
]

# Citations found in the paper
cited = [
    'Ainslie2023', 'Beltagy2020', 'Chen2018', 'Dai2019', 'Dao2023', 'Deng2025',
    'EAGLE2024', 'Fazlyab2019', 'Geifman2017', 'Goyal2021', 'Kwon2023',
    'Li2024lookahead', 'Liu2025a', 'Liu2025b', 'LopezPaz2015', 'Miyato2018',
    'PSLM2024', 'Ren2023', 'Rodionov2025', 'Stern2018', 'Vapnik2015',
    'Wei2025sidechannel', 'Xiao2025sprint', 'Yan2024', 'Yoshikawa2023',
    'Zheng2025', 'bachlechner2021rezero', 'bulatov2022recurrent', 'cai2024medusa',
    'gu2023mamba', 'houlsby2019parameter', 'hu2021lora', 'leviathan2023fast',
    'ning2023skeleton', 'snc_code', 'survey2025'
]

cited_set = set(cited)
bib_set = set(bib_entries)

orphaned = sorted(bib_set - cited_set)
duplicates = []

# Check for duplicates (same paper, different key)
if 'Medusa2024' in bib_set and 'cai2024medusa' in cited_set:
    duplicates.append('Medusa2024 (duplicate of cai2024medusa)')
if 'Ning2023' in bib_set and 'ning2023skeleton' in cited_set:
    duplicates.append('Ning2023 (duplicate of ning2023skeleton)')

print("="*70)
print("CITATION ANALYSIS")
print("="*70)
print(f"\nTotal bibliography entries: {len(bib_entries)}")
print(f"Total citations used: {len(cited_set)}")
print(f"Orphaned entries: {len(orphaned)}")

if orphaned:
    print(f"\n❌ ORPHANED ENTRIES TO REMOVE ({len(orphaned)}):")
    for entry in orphaned:
        print(f"   - {entry}")

if duplicates:
    print(f"\n⚠️  DUPLICATE ENTRIES TO REMOVE ({len(duplicates)}):")
    for dup in duplicates:
        print(f"   - {dup}")

print(f"\n✅ PROPERLY CITED ENTRIES ({len(cited_set)}):")
for entry in sorted(cited_set):
    print(f"   ✓ {entry}")

print("\n" + "="*70)


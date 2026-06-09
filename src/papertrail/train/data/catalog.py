"""Realistic sample pools for synthetic document generation.

Pure data, no logic. Generators draw from these with a seeded RNG so the whole
dataset is reproducible from a single `--seed`.
"""

from __future__ import annotations

# --- Pharmacies & medications -------------------------------------------------

PHARMACIES = [
    "CVS Pharmacy",
    "Walgreens",
    "Rite Aid",
    "Costco Pharmacy",
    "Walmart Pharmacy",
    "Kroger Pharmacy",
    "City Drug & Wellness",
    "Northgate Pharmacy",
    "HealthMart Pharmacy",
    "Safeway Pharmacy",
]

# Prescription drugs (HSA-eligible).
RX_DRUGS = [
    ("Lisinopril 10mg", "30 tablets"),
    ("Atorvastatin 20mg", "90 tablets"),
    ("Metformin 500mg", "60 tablets"),
    ("Amoxicillin 500mg", "21 capsules"),
    ("Albuterol HFA", "1 inhaler"),
    ("Sertraline 50mg", "30 tablets"),
    ("Omeprazole 20mg", "30 capsules"),
    ("Levothyroxine 75mcg", "90 tablets"),
]

# OTC drugs — HSA-eligible WITHOUT prescription post-CARES Act 2020.
OTC_DRUGS = [
    ("Ibuprofen 200mg", "100 ct"),
    ("Acetaminophen 500mg", "50 ct"),
    ("Loratadine 10mg", "30 ct"),
    ("Omeprazole OTC 20mg", "42 ct"),
    ("Cetirizine 10mg", "30 ct"),
    ("Aspirin 81mg", "120 ct"),
]

# Items that are NOT HSA-eligible — mixed onto receipts as distractors.
NON_MEDICAL_ITEMS = [
    ("Greeting Card", 4.99),
    ("Bag of Chips", 3.49),
    ("Magazine", 6.99),
    ("Bottled Water 24pk", 5.99),
    ("Phone Charger", 14.99),
    ("Candy Bar", 1.79),
]

# --- Dental -------------------------------------------------------------------

DENTAL_OFFICES = [
    "Bright Smile Dental",
    "City Dental Group",
    "Lakeside Family Dentistry",
    "Gentle Care Dental",
    "Summit Orthodontics",
    "Maple Street Dental",
]
DENTAL_PROCEDURES = [
    ("Routine Cleaning (D1110)", 120.00),
    ("Bitewing X-Rays (D0274)", 65.00),
    ("Composite Filling (D2391)", 185.00),
    ("Crown - Porcelain (D2740)", 1100.00),
    ("Periodic Exam (D0120)", 55.00),
    ("Fluoride Treatment (D1206)", 35.00),
]

# --- Vision -------------------------------------------------------------------

VISION_OFFICES = [
    "ClearView Optometry",
    "20/20 Eye Care",
    "Visionworks",
    "LensCrafters",
    "Family Eye Associates",
    "Pearle Vision",
]
VISION_ITEMS = [
    ("Comprehensive Eye Exam", 95.00),
    ("Single Vision Lenses", 120.00),
    ("Progressive Lenses", 280.00),
    ("Frames", 165.00),
    ("Contact Lens Fitting", 75.00),
    ("Box of Contacts (90 day)", 110.00),
]

# --- Doctor / medical ---------------------------------------------------------

CLINICS = [
    "Riverside Family Medicine",
    "Mercy Urgent Care",
    "Downtown Medical Associates",
    "Pinecrest Pediatrics",
    "Wellspring Internal Medicine",
    "Cornerstone Clinic",
]
MEDICAL_SERVICES = [
    ("Office Visit - Established (99213)", 145.00),
    ("Office Visit - New (99203)", 210.00),
    ("Venipuncture (36415)", 18.00),
    ("Comprehensive Metabolic Panel (80053)", 65.00),
    ("Flu Vaccine (90686)", 40.00),
    ("EKG (93000)", 85.00),
    ("Specialist Copay", 50.00),
    ("Urgent Care Visit", 175.00),
]

# --- Insurers (for EOBs) ------------------------------------------------------

INSURERS = [
    "Blue Cross Blue Shield",
    "Aetna",
    "UnitedHealthcare",
    "Cigna",
    "Kaiser Permanente",
    "Humana",
]

# --- Ineligible-only documents ------------------------------------------------

INELIGIBLE_MERCHANTS = [
    "Gold's Gym",
    "Planet Fitness",
    "GNC Supplements",
    "Sephora",
    "Lush Cosmetics",
    "Anytime Fitness",
]
INELIGIBLE_ITEMS = [
    ("Monthly Gym Membership", 49.99),
    ("Personal Training Session", 75.00),
    ("Multivitamin Supplement", 24.99),
    ("Whey Protein 2lb", 39.99),
    ("Anti-Aging Face Cream", 89.00),
    ("Teeth Whitening Kit (cosmetic)", 45.00),
    ("Collagen Powder", 34.99),
]

# --- People & places ----------------------------------------------------------

PATIENTS = [
    "John Smith",
    "Maria Garcia",
    "David Chen",
    "Sarah Johnson",
    "Michael Brown",
    "Emily Davis",
    "James Wilson",
    "Linda Martinez",
    "Robert Taylor",
    "Jennifer Lee",
]

STREETS = [
    "142 Oak Avenue",
    "87 Maple Street",
    "2200 Commerce Blvd",
    "55 Lakeshore Dr",
    "913 Elm Street",
    "476 Sunset Way",
]
CITIES = [
    ("Springfield", "IL", "62704"),
    ("Portland", "OR", "97204"),
    ("Austin", "TX", "78701"),
    ("Denver", "CO", "80202"),
    ("Columbus", "OH", "43215"),
    ("Raleigh", "NC", "27601"),
]

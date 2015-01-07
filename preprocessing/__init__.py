from collections import OrderedDict


float_columns = {
    'FTE',
    'Total'
}


text_columns = {
    'Facility_or_Department',
    'Function_Description',
    'Fund_Description',
    'Job_Title_Description',
    'Location_Description',
    'Object_Description',
    'Position_Extra',
    'Program_Description',
    'SubFund_Description',
    'Sub_Object_Description',
    'Text_1',
    'Text_2',
    'Text_3',
    'Text_4'
}


multi_labels = OrderedDict([
    ('Function', [
        'Aides Compensation',
        'Career & Academic Counseling',
        'Communications',
        'Curriculum Development',
        'Data Processing & Information Services',
        'Development & Fundraising',
        'Enrichment',
        'Extended Time & Tutoring',
        'Facilities & Maintenance',
        'Facilities Planning',
        'Finance, Budget, Purchasing & Distribution',
        'Food Services',
        'Governance',
        'Human Resources',
        'Instructional Materials & Supplies',
        'Insurance',
        'Legal',
        'Library & Media',
        'NO_LABEL',
        'Other Compensation',
        'Other Non-Compensation',
        'Parent & Community Relations',
        'Physical Health & Services',
        'Professional Development',
        'Recruitment',
        'Research & Accountability',
        'School Administration',
        'School Supervision',
        'Security & Safety',
        'Social & Emotional',
        'Special Population Program Management & Support',
        'Student Assignment',
        'Student Transportation',
        'Substitute Compensation',
        'Teacher Compensation',
        'Untracked Budget Set-Aside',
        'Utilities'
    ]),
     ('Object_Type', [
        'Base Salary/Compensation',
        'Benefits',
        'Contracted Services',
        'Equipment & Equipment Lease',
        'NO_LABEL',
        'Other Compensation/Stipend',
        'Other Non-Compensation',
        'Rent/Utilities',
        'Substitute Compensation',
        'Supplies/Materials',
        'Travel & Conferences'
    ]),
    ('Operating_Status', [
        'Non-Operating',
        'Operating, Not PreK-12',
        'PreK-12 Operating'
    ]),
    ('Position_Type', [
        '(Exec) Director',
        'Area Officers',
        'Club Advisor/Coach',
        'Coordinator/Manager',
        'Custodian',
        'Guidance Counselor',
        'Instructional Coach',
        'Librarian',
        'NO_LABEL',
        'Non-Position',
        'Nurse',
        'Nurse Aide',
        'Occupational Therapist',
        'Other',
        'Physical Therapist',
        'Principal',
        'Psychologist',
        'School Monitor/Security',
        'Sec/Clerk/Other Admin',
        'Social Worker',
        'Speech Therapist',
        'Substitute',
        'TA',
        'Teacher',
        'Vice Principal'
    ]),
     ('Pre_K', [
        'NO_LABEL',
        'Non PreK',
        'PreK'
    ]),
    ('Reporting', [
        'NO_LABEL',
        'Non-School',
        'School'
    ]),
    ('Sharing', [
        'Leadership & Management',
        'NO_LABEL',
        'School Reported',
        'School on Central Budgets',
        'Shared Services'
    ]),
    ('Student_Type', [
        'Alternative',
        'At Risk',
        'ELL',
        'Gifted',
        'NO_LABEL',
        'Poverty',
        'PreK',
        'Special Education',
        'Unspecified'
    ]),
    ('Use', [
        'Business Services',
        'ISPD',
        'Instruction',
        'Leadership',
        'NO_LABEL',
        'O&M',
        'Pupil Services & Enrichment',
        'Untracked Budget Set-Aside'
    ])
])
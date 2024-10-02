# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "" ""
__status__ = "Development"

import utilpackage.index_map as index_map

ACE_ENTITY_TYPES = [
    'GPE:GPE-Cluster',
    'GPE:Nation',
    'GPE:Population-Center',
    'VEH:Underspecified',
    'WEA:Projectile',
    'FAC:Subarea-Facility',
    'WEA:Chemical',
    'LOC:Region-General',
    'ORG:Entertainment',
    'Contact-Info:URL',
    'WEA:Biological',
    'ORG:Medical-Science',
    'ORG:Educational',
    'Numeric:Percent',
    'LOC:Region-International',
    'WEA:Shooting',
    'GPE:State-or-Province',
    'VEH:Land',
    'Crime',
    'ORG:Government',
    'PER:Group',
    'Job-Title',
    'Contact-Info:E-Mail',
    'LOC:Celestial',
    'GPE:Special',
    'ORG:Media',
    'WEA:Nuclear',
    'Numeric:Money',
    'FAC:Plant',
    'WEA:Exploding',
    'LOC:Water-Body',
    'ORG:Sports',
    'FAC:Airport',
    'GPE:County-or-District',
    'PER:Indeterminate',
    'GPE:Continent',
    'FAC:Building-Grounds',
    'VEH:Subarea-Vehicle',
    'ORG:Religious',
    'ORG:Commercial',
    'FAC:Path',
    'LOC:Boundary',
    'Sentence',
    'WEA:Underspecified',
    'Contact-Info:Phone-Number',
    'PER:Individual',
    'VEH:Air',
    'ORG:Non-Governmental',
    'VEH:Water',
    'LOC:Land-Region-Natural',
    'WEA:Blunt',
    'WEA:Sharp',
    'LOC:Address',
    'O'
]
"""list: A list of golden entity types in ACE2005."""

ACE_ENTITY_TYPES_MAP = index_map.IndexMap(ACE_ENTITY_TYPES)
""":class::`index_map.IndexMap`: ACE entity types mapped to indices."""

ACE_RELATION_TYPES_FINE_GRAINED = [
    'PHYS:Near',
    'PER-SOC:Family',
    'ORG-AFF:Sports-Affiliation',
    'PER-SOC:Business',
    'ORG-AFF:Employment',
    'PART-WHOLE:Geographical',
    'GEN-AFF:Citizen-Resident-Religion-Ethnicity',
    'ORG-AFF:Membership',
    'ORG-AFF:Investor-Shareholder',
    'ORG-AFF:Ownership',
    'PART-WHOLE:Artifact',
    'PER-SOC:Lasting-Personal',
    'ORG-AFF:Founder',
    'ORG-AFF:Student-Alum',
    'PHYS:Located',
    'GEN-AFF:Org-Location',
    'ART:User-Owner-Inventor-Manufacturer',
    'PART-WHOLE:Subsidiary'
]
"""list: A list of fine-grained golden relation types in ACE2005."""

ACE_RELATION_TYPES = ['PER-SOC', 'ART', 'PHYS', 'PART-WHOLE', 'ORG-AFF', 'GEN-AFF']
"""list: A list of  golden relation types in ACE2005."""

ACE_RELATION_TYPES_MAP = index_map.IndexMap(ACE_RELATION_TYPES)

ACE_RELATION_TYPES_FINE_GRAINED_MAP = index_map.IndexMap(ACE_RELATION_TYPES_FINE_GRAINED)
""":class::`index_map.IndexMap`: ACE relation types mapped to indices."""

ALBERT_XX_LARGE_VERSION = "albert-xxlarge-v1"  # "albert-base-v2"
"""str: The version of ALBERT."""

BERT_BASE_VERSION = "bert-base-uncased"
"""str: The version of BERT."""

BEST_CKPT_FILE = "best.ckpt"
"""str: The filename of the stored checkpoint of the best model."""

COARSE_GRAINED_ACE_ENTITY_TYPES = [
    'B-Contact-Info',
    'B-Crime',
    'B-FAC',
    'B-GPE',
    'B-Job-Title',
    'B-LOC',
    'B-Numeric',
    'B-ORG',
    'B-PER',
    'B-Sentence',
    'B-VEH',
    'B-WEA',
    'I-Contact-Info',
    'I-Crime',
    'I-FAC',
    'I-GPE',
    'I-Job-Title',
    'I-LOC',
    'I-Numeric',
    'I-ORG',
    'I-PER',
    'I-Sentence',
    'I-VEH',
    'I-WEA',
    'O'
]
"""list: The list of coarse grained entity types tag in ACE2005."""

COARSE_GRAINED_ACE_ENTITY_TYPES_MAP = index_map.IndexMap(COARSE_GRAINED_ACE_ENTITY_TYPES)
""":class::`index_map.IndexMap`: Coarse grained ACE2005 entity types mapped to indices."""

DATASET_DIRS = {
    "nyt": "nyt/",
    "pretrain": "pretrain/",
    "retacred": "retacred",
    "tacred": "tacred",
    "tacrev": "tacrev",
    "wikidata": "wikidata/"
}
"""dict: A mapping of datasets to their directories."""

NYT_RELATION_TYPES = [
    '/broadcast/content/location',
    '/broadcast/producer/location',
    '/business/business_location/parent_company',
    '/business/company/advisors',
    '/business/company/founders',
    '/business/company/industry',
    '/business/company/locations',
    '/business/company/major_shareholders',
    '/business/company/place_founded',
    '/business/company_advisor/companies_advised',
    '/business/company_shareholder/major_shareholder_of',
    '/business/person/company',
    '/business/shopping_center/owner',
    '/business/shopping_center_owner/shopping_centers_owned',
    '/film/film/featured_film_locations',
    '/film/film_festival/location',
    '/film/film_location/featured_in_films',
    '/location/administrative_division/country',
    '/location/br_state/capital',
    '/location/cn_province/capital',
    '/location/country/administrative_divisions',
    '/location/country/capital',
    '/location/de_state/capital',
    '/location/fr_region/capital',
    '/location/in_state/administrative_capital',
    '/location/in_state/judicial_capital',
    '/location/in_state/legislative_capital',
    '/location/it_region/capital',
    '/location/jp_prefecture/capital',
    '/location/location/contains',
    '/location/mx_state/capital',
    '/location/neighborhood/neighborhood_of',
    '/location/province/capital',
    '/location/us_county/county_seat',
    '/location/us_state/capital',
    '/people/deceased_person/place_of_burial',
    '/people/deceased_person/place_of_death',
    '/people/ethnicity/geographic_distribution',
    '/people/ethnicity/included_in_group',
    '/people/ethnicity/includes_groups',
    '/people/ethnicity/people',
    '/people/family/country',
    '/people/family/members',
    '/people/person/children',
    '/people/person/ethnicity',
    '/people/person/nationality',
    '/people/person/place_lived',
    '/people/person/place_of_birth',
    '/people/person/profession',
    '/people/person/religion',
    '/people/place_of_interment/interred_here',
    '/people/profession/people_with_this_profession',
    '/sports/sports_team/location',
    '/sports/sports_team_location/teams',
    '/time/event/locations',
    'NA'
]
"""list: The relation types in the NYT dataset."""

NYT_RELATION_TYPES_MAP = index_map.IndexMap(NYT_RELATION_TYPES)

REL_ARG_ONE_BEGIN = "B-Arg1"
"""str: The tag indicating the beginning of argument one in the relation."""

REL_ARG_ONE_INTERMEDIATE = "I-Arg1"
"""str: The tag indicating the intermediate positions of argument one in the relation."""

REL_ARG_TWO_BEGIN = "B-Arg2"
"""str: The tag indicating the beginning of argument two in the relation."""

REL_ARG_TWO_INTERMEDIATE = "I-Arg2"
"""str: The tag indicating the intermediate positions of argument two in the relation."""

RELATION_ENTITY_ROLE_LABELS = ['B-Arg1', 'B-Arg2', 'I-ARG-1', 'I-ARG-2', 'O']
"""list: The list of labels for the roles the entities are playing in a given relation."""

RELATION_ENTITY_ROLE_LABELS_MAP = index_map.IndexMap(RELATION_ENTITY_ROLE_LABELS)

RETACRED_RELATION_TYPES = [
    'no_relation',
    'org:alternate_names',
    'org:city_of_branch',
    'org:country_of_branch',
    'org:dissolved',
    'org:founded',
    'org:founded_by',
    'org:member_of',
    'org:members',
    'org:number_of_employees/members',
    'org:political/religious_affiliation',
    'org:shareholders',
    'org:stateorprovince_of_branch',
    'org:top_members/employees',
    'org:website',
    'per:age',
    'per:cause_of_death',
    'per:charges',
    'per:children',
    'per:cities_of_residence',
    'per:city_of_birth',
    'per:city_of_death',
    'per:countries_of_residence',
    'per:country_of_birth',
    'per:country_of_death',
    'per:date_of_birth',
    'per:date_of_death',
    'per:employee_of',
    'per:identity',
    'per:origin',
    'per:other_family',
    'per:parents',
    'per:religion',
    'per:schools_attended',
    'per:siblings',
    'per:spouse',
    'per:stateorprovince_of_birth',
    'per:stateorprovince_of_death',
    'per:stateorprovinces_of_residence',
    'per:title'
]
"""list: The relation types in the Re-Tacred dataset."""

RETACRED_RELATION_TYPES_MAP = index_map.IndexMap(RETACRED_RELATION_TYPES)

ROBERTA_LARGE_VERSION = "roberta-large"  # "roberta-base"
"""str: The version of RoBERTa"""

TACRED_ENTITY_TYPES = [
    'CAUSE_OF_DEATH',
    'CITY',
    'COUNTRY',
    'CRIMINAL_CHARGE',
    'DATE',
    'DURATION',
    'IDEOLOGY',
    'LOCATION',
    'MISC',
    'NATIONALITY',
    'NUMBER',
    'ORGANIZATION',
    'PERSON',
    'RELIGION',
    'STATE_OR_PROVINCE',
    'TITLE',
    'URL'
]
"""list: The entity types in TACRED."""

TACRED_ENTITY_TYPES_MAP = index_map.IndexMap(TACRED_ENTITY_TYPES)
""":class::`index_map.IndexMap`: A map of Tacred entity types to indices."""

TACRED_RELATION_TYPES = [
    'no_relation',
    'org:alternate_names',
    'org:city_of_headquarters',
    'org:country_of_headquarters',
    'org:dissolved',
    'org:founded',
    'org:founded_by',
    'org:member_of',
    'org:members',
    'org:number_of_employees/members',
    'org:parents',
    'org:political/religious_affiliation',
    'org:shareholders',
    'org:stateorprovince_of_headquarters',
    'org:subsidiaries',
    'org:top_members/employees',
    'org:website',
    'per:age',
    'per:alternate_names',
    'per:cause_of_death',
    'per:charges',
    'per:children',
    'per:cities_of_residence',
    'per:city_of_birth',
    'per:city_of_death',
    'per:countries_of_residence',
    'per:country_of_birth',
    'per:country_of_death',
    'per:date_of_birth',
    'per:date_of_death',
    'per:employee_of',
    'per:origin',
    'per:other_family',
    'per:parents',
    'per:religion',
    'per:schools_attended',
    'per:siblings',
    'per:spouse',
    'per:stateorprovince_of_birth',
    'per:stateorprovince_of_death',
    'per:stateorprovinces_of_residence',
    'per:title'
]
"""list: The relation types in TACRED Dataset."""

TACRED_RELATION_TYPES_MAP = index_map.IndexMap(TACRED_RELATION_TYPES)
""":class::`index_map.IndexMap`: A map of Tacred relation types to indices."""

OTHER_TAG = "O"
"""str: The tag indicating the token is none of the defined labels/tags."""

WIKIDATA_RELATION_TYPES = [
    'P0',
    'P1001',
    'P101', 'P1018', 'P102', 'P1027', 'P103', 'P1038', 'P1040', 'P1049', 'P1050', 'P1056', 'P1057', 'P106',
    'P1064', 'P1068', 'P1071', 'P1072', 'P1074', 'P108', 'P1080', 'P1081', 'P110', 'P111', 'P112', 'P113', 'P114',
    'P1142', 'P115', 'P1158', 'P1165', 'P118', 'P1181', 'P119', 'P1191', 'P1192', 'P1196', 'P121', 'P122', 'P123',
    'P1249', 'P126', 'P1268', 'P1269', 'P127', 'P1302', 'P1303', 'P1308', 'P131', 'P1312', 'P1313', 'P1317', 'P1322',
    'P1327', 'P1336', 'P134', 'P1344', 'P1346', 'P135', 'P136', 'P1363', 'P1365', 'P138', 'P1382', 'P1383', 'P1389',
    'P1399', 'P140', 'P1408', 'P141', 'P1411', 'P1412', 'P1414', 'P1416', 'P1420', 'P1429', 'P1431', 'P1433', 'P1434',
    'P1435', 'P144', 'P1455', 'P1462', 'P1478', 'P1479', 'P149', 'P150', 'P1531', 'P1532', 'P1535', 'P1547', 'P155',
    'P1552', 'P1557', 'P1560', 'P157', 'P1571', 'P1574', 'P1582', 'P1589', 'P159', 'P16', 'P161', 'P1619', 'P162',
    'P1622', 'P163', 'P1654', 'P166', 'P169', 'P17', 'P170', 'P172', 'P1731', 'P175', 'P176', 'P177', 'P178', 'P179',
    'P180', 'P1809', 'P1811', 'P183', 'P184', 'P1851', 'P186', 'P1875', 'P1885', 'P1889', 'P189', 'P1891', 'P19',
    'P190', 'P1906', 'P194', 'P195', 'P196', 'P1962', 'P197', 'P199', 'P1990', 'P1995', 'P20', 'P200', 'P201', 'P205',
    'P206', 'P2079', 'P208', 'P209', 'P2098', 'P21', 'P2152', 'P2155', 'P2184', 'P22', 'P2286', 'P2289', 'P2293',
    'P2341', 'P2348', 'P237', 'P2389', 'P241', 'P2416', 'P25', 'P2505', 'P2512', 'P2541', 'P2546', 'P2578', 'P2596',
    'P26', 'P263', 'P2632', 'P2633', 'P264', 'P2670', 'P27', 'P272', 'P275', 'P276', 'P277', 'P279', 'P282', 'P286',
    'P287', 'P289', 'P291', 'P30', 'P306', 'P31', 'P344', 'P35', 'P355', 'P36', 'P360', 'P361', 'P364', 'P366', 'P37',
    'P371', 'P375', 'P376', 'P38', 'P39', 'P397', 'P399', 'P40', 'P400', 'P403', 'P406', 'P407', 'P408', 'P410', 'P411',
    'P412', 'P413', 'P414', 'P415', 'P417', 'P421', 'P425', 'P427', 'P437', 'P449', 'P450', 'P451', 'P452', 'P457',
    'P460', 'P461', 'P462', 'P463', 'P466', 'P467', 'P469', 'P47', 'P485', 'P495', 'P50', 'P500', 'P501', 'P504',
    'P509', 'P512', 'P516', 'P517', 'P522', 'P523', 'P53', 'P530', 'P533', 'P54', 'P541', 'P547', 'P551', 'P553',
    'P559', 'P562', 'P567', 'P569', 'P57', 'P570', 'P571', 'P575', 'P577', 'P58', 'P59', 'P6', 'P607', 'P609', 'P61',
    'P610', 'P611', 'P612', 'P618', 'P619', 'P620', 'P629', 'P641', 'P647', 'P655', 'P658', 'P66', 'P664', 'P669',
    'P674', 'P676', 'P681', 'P684', 'P688', 'P689', 'P69', 'P7', 'P703', 'P706', 'P708', 'P725', 'P726', 'P729', 'P730',
    'P734', 'P735', 'P737', 'P740', 'P748', 'P750', 'P765', 'P767', 'P768', 'P769', 'P770', 'P780', 'P788', 'P790',
    'P793', 'P800', 'P802', 'P807', 'P81', 'P822', 'P825', 'P826', 'P828', 'P831', 'P832', 'P84', 'P840', 'P85', 'P86',
    'P868', 'P87', 'P870', 'P88', 'P880', 'P885', 'P9', 'P91', 'P913', 'P915', 'P92', 'P921', 'P927', 'P931', 'P937',
    'P941', 'P945', 'P97', 'P972', 'P98', 'P991'
]

WIKIDATA_RELATION_TYPES_MAP = index_map.IndexMap(WIKIDATA_RELATION_TYPES)

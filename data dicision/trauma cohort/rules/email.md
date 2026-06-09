email 1 attach: Trauma Registry Inclusion and Exclusion Criteria.docx

The National Trauma Data Standard (NTDS) has a list of ICD 10 CM codes that are used to define trauma registry inclusion criteria which I have pasted below. There are additional criteria which makes it difficult for us to automate the patient identification process, so registry patient identification is a manual process for us. I’ve attached our inclusion/exclusion criteria which includes state registry inclusion criteria, burn registry inclusion criteria, and Harborview inclusion criteria (primarily all initial firearm injuries are included). If you’re interested, I can send a link to an Epic workbench report that queries for patients that meet ACS/NTDS Trauma Registry inclusion criteria. The report works well, but only queries patients that come through the ED with a specific Encounter Diagnosis Grouper and doesn’t factor in some of the other inclusion criteria we follow.

 

 

***International Classification of Diseases, Tenth Revision\*** **(ICD-10-CM):**

• S00-S99 with 7th character modifiers of A, B, or C ONLY. (Injuries to specific body parts–initial encounter)

• T07 (unspecified multiple injuries)

• T14 (injury of unspecified body region)

• T79.A1-T79.A9 with 7th character modifier of A ONLY (Traumatic Compartment Syndrome–initial encounter)

 

**EXCLUDING the following isolated injuries:**

***ICD-10-CM:\***

• S00 (Superficial injuries of the head)

• S10 (Superficial injuries of the neck)

• S20 (Superficial injuries of the thorax)

• S30 (Superficial injuries of the abdomen, pelvis, lower back and external genitals)

• S40 (Superficial injuries of shoulder and upper arm)

• S50 (Superficial injuries of elbow and forearm)

• S60 (Superficial injuries of wrist, hand and fingers)

• S70 (Superficial injuries of hip and thigh)

• S80 (Superficial injuries of knee and lower leg)

• S90 (Superficial injuries of ankle, foot and toes)



email 2: attach: qualified_traumatic_EcodesGOK.xlsx, email 1

I’ve added an “exclude” column to the file. Also attached is the reply from our trauma registry lead. In summary, trauma registries don’t use E-codes for inclusion, but actual injuries sustained. The E-codes are included in the registry datasets for epidemiology, etc.

 

I would suggest using E-codes to extract cases from MIMIC IV as the first step. The second step will be to exclude cases that don’t have a calculated injury severity score (ISS) or any of the AIS components.
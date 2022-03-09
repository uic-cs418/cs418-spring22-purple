# cs418-spring22-purple


# Gun Violence in the US
# Motivation

Big Idea: Explore the correlation between gun violence cases with age, location, and/or socioeconomic factors in the area.
Our Problem: Which groups have an increase in gun violence, and is there any bias with the victims?

Question We Want To Answer: Has gun reform impacted the number of cases in the US?


Significance of the Topic: Gun violence is a very common threat to all humans and begins to impact more people as years go on. It affects everyone, however, we have seen in recent news that certain groups, such as, people of color or people from locations that don’t have as much money.

Hypothesis: Gun violence decreases after gun reform.


# Dataset Information
Dataset: https://github.com/jamesqo/gun-violence-data

- Data is stored in a .csv file ordered by ascending date from January 2013 to March 2018 and has no missing dates.
- Data Type: Number of gun incidents and their details between 2013-2018
- Data Size:  239, 678 rows and 29 columns
- Feature Types: int, string, boolean, list[string], dict[int, string], or float

- Additional Information:
    Each field has a specified format (i.e. address field only provides address of where the incident took place)
    Not all fields are required. We will use categories, such as, incident_id, date, state, city_or_county, etc. Additionally, we may drop any unneeded columns.

# Project Goals
Approach: Aggregate Gun Violence dataset with 3 datasets on gun reform laws, demographics, and population

Scope of Project: Visualize the number of gun cases in the US using the Gun Violence dataset from the Gun Violence Archive.

Hypothesized End Results: Find the relationship between socio-economic factors vs gun violence cases, categorize gun violence incidents by demographics

What we hope to achieve for progress report: Categorize dataset based on whether the incident occurred before or after gun reform. Explore machine learning applications to predict likelihood of gun incidents in response to gun reform
We plan to have a static system.


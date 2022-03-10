# Gun Violence in the US
cs418-spring22-purple
# Motivation

***Big Idea***: Explore the correlation between gun violence cases with age, location, and/or socioeconomic factors in the area.
***Our Problem***: Which groups have an increase in gun violence, and is there any bias with the victims?

***Question We Want To Answer***: Has gun reform impacted the number of gun violence cases in the US?


***Significance***: Gun violence is a very common threat to all humans and begins to impact more people as years go on. It affects everyone, however, some groups are impacted much more than others, such as, people of color or people from locations that don’t have as much money.

***Hypothesis***: Gun violence decreases after more gun reform laws are passed.


# Dataset Information
***Dataset***: https://github.com/jamesqo/gun-violence-data

* Data is stored in a .csv file ordered by ascending date from January 2013 to March 2018. This dataset has no missing dates.
* Data Type: Number of gun incidents and their description details between 2013 and 2018
* Data Size:  239,678 rows and 29 columns
* Feature Types: int, string, boolean, list[string], dict[int, string], or float

***Additional Information***:
* Each field has a specified format (i.e. address field only provides address of where the incident took place)
* Not all fields are required. We will use categories, such as, incident_id, date, state, city_or_county, etc. Additionally, we may drop any unneeded columns.

# Project Goals
***Approach***: Aggregate the Gun Violence dataset with 3 additional datasets on gun reform laws, demographics, and population.

***Scope of Project***: Visualize the number of gun cases in the US using the Gun Violence dataset from the Gun Violence Archive.

***Hypothesized End Results***:
    * Find the relationship between socio-economic factors and gun violence cases
    * Categorize the number of gun violence incidents by demographics

***Goals by Progress Report***:
    * Categorize the dataset based on whether the gun violence incident occurred before or after gun reform.
    * Explore machine learning applications to predict likelihood of gun incidents in response to gun reform

We plan to have a static system.


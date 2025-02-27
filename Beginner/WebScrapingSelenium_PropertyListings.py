'''
18/2/2025 Finishing learning some concepts of webscraping in order to get data. Data are collected from a website with list instead of prepared tabular data.
        Selenium webdriver are replaced with undetected_chromedriver for easy scraping on multiple pages in order to bypass the website security human verification.
        Data gathered are allowed in https://www.propertyguru.com.my/bm/rumah-banglo-dijual/robots.txt . No harm were made throughout the learning process lol

        Let's go to process the data for machine learning next! by : FxHakkun
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
import json
import pandas as pd
import time

#Set up the Selenium WebDriver 
driver = uc.Chrome()
#Open URL and send GET
base_url = "https://www.propertyguru.com.my/bm/rumah-banglo-dijual"
property_data = []

for page in range(1,101): #take first 100 pages data
    #condition for url fitting
    if page == 1:
        url = base_url
    else:
        url = f"{base_url}/{page}"
    driver.get(url)
    #adding delay for page to load
    try:
        WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH, '//script[@id="__NEXT_DATA__"]')))
    except Exception as e:
        print(f"Error loading page {page}: {e}")
        continue
    #Locate the <script> tag with the JSON Data
    script_tag = driver.find_element(By.XPATH, '//script[@id="__NEXT_DATA__"]')
    #Extract the JSON Content
    json_data = script_tag.get_attribute("innerHTML")
    #Parse the JSON Data
    data = json.loads(json_data)

    listings = data["props"]["pageProps"]["pageData"]["data"]["listingsData"]

    for listing in listings:
        listing_data = listing["listingData"]

        #Extract relevant fields
        name = listing_data.get("localizedTitle", "N/A")
        price = listing_data["price"].get("pretty","N/A")
        price_perArea = listing_data.get("pricePerArea", {}).get("localeStringValue", "N/A")
        if len(listing_data["listingFeatures"]) > 1:
            size_floor = listing_data["listingFeatures"][1][0].get("text", "N/A") 
        else :
            size_floor = "N/A"
        size_land = listing_data["area"].get("localeStringValue", "N/A")
        location = listing_data.get("fullAddress", "N/A")
        bedrooms = listing_data["listingFeatures"][0][0].get("text", "N/A") if len(listing_data["listingFeatures"][0]) > 0 else "N/A"
        bathrooms = listing_data["listingFeatures"][0][1].get("text", "N/A") if len(listing_data["listingFeatures"][0]) > 1 else "N/A"
        property_type = listing_data["property"].get("typeText", "N/A")
        url = listing_data.get("url","N/A")

        #Append data to the list
        property_data.append({
            "Name" : name,
            "Price" : price,
            "Price (Per Area)" : price_perArea,
            "Size (Floor)" : size_floor,
            "Size (Land)" : size_land,
            "Location" : location,
            "Bedrooms" : bedrooms,
            "Bathrooms" : bathrooms,
            "Type of Property" : property_type,
            "URL" : url
        })
    #Delay between each pages
    time.sleep(2)

#Close the browser
try:
    driver.quit()
except OSError:
    pass
#Convert to DataFrame
df = pd.DataFrame(property_data)
df.to_csv("property_listings.csv", index= False)
print("Data successfully saved to CSV files!")

'''
#Check the keys in the JSON
print(data.keys())
print(json.dumps(data, indent=2))
df = pd.DataFrame([json.dumps(data,indent=2)])
df.to_csv('JSON_DUMPS.csv', index=False)
'''

'''
if response.status_code == 200:
    #Parse the content using BeautifulSoup
    print("Successful")
    soup = BeautifulSoup(response.content, 'html.parser')

    #Find all property listing
    listings = soup.find_all('div',class_='listing-card')

    data = [] #to store scraped data

    #loop through each of the listings and extract details
    for listing in listings:
        #Extract property name
        name = listing.find('h2', class_='property-name'.text.strip())
        #Extract property price
        price = listing.find('div', class_='price'.text.strip())
        #Extract property location
        location = listing.find('div', class_='location'.text.strip())
        #Append data to the list data[]
        data.append({
            'Name' : name,
            'Price' : price,
            'Location' : location
            })
    """#Convert list to Dataframe
    df = pd.DataFrame(data)
    #Save the data to CSV file
    df.to_csv('property_listings.csv', index = False)
    print("Data saved to property_listings.csv")"""

    print(data)

else:
    #possible error if exist
    print(f"Failed to retrieve the webpage. Status code : {response.status_code}")
    '''


from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import json
import pandas as pd
import time

#Set up the Selenium WebDriver
driver = webdriver.Chrome()
#Open URL and send GET
url = "https://www.propertyguru.com.my/bm/rumah-banglo-dijual"
driver.get(url)
#adding delay for page to load
time.sleep(5)
#Locate the <script> tag with the JSON Data
script_tag = driver.find_element(By.XPATH, '//script[@id="__NEXT_DATA__"]')
#Extract the JSON Content
json_data = script_tag.get_attribute("innerHTML")
#Parse the JSON Data
data = json.loads(json_data)
#Close the browser
driver.quit()

listings = data["props"]["pageProps"]["pageData"]["data"]["listingsData"]

property_data = []

for listing in listings:
    listing_data = listing["listingData"]

    #Extract relevant fields
    name = listing_data.get("localizedTitle", "N/A")
    price = listing_data["price"].get("pretty","N/A")
    price_perArea = listing_data.get("pricePerArea", {}).get("localeStringValue", "N/A")
    size_floor = listing_data["listingFeatures"][1][0].get("text", "N/A") 
    size_land = listing_data["area"].get("localeStringValue", "N/A")
    location = listing_data.get("fullAddress", "N/A")
    bedrooms = listing_data["listingFeatures"][0][0].get("text", "N/A")
    bathrooms = listing_data["listingFeatures"][0][1].get("text", "N/A")
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


#DATA COLLECTION AND WEB SCRAPING

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# 1. Configuration & Target URL
BASE_URL = "https://quotes.toscrape.com"
START_PAGE = "/page/1/"


def scrape_quotes():
    all_quotes = []
    current_page = START_PAGE

    # Custom User-Agent header makes your script look like a standard web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print("Starting data collection...")

    while current_page:
        url = f"{BASE_URL}{current_page}"
        print(f"Scraping: {url}")

        # Fetch the page content
        try:
            response = requests.get(url, headers=headers, timeout=10)
            # Raise an error if the request failed (e.g., 404, 500)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            break

        # Parse the HTML structure
        soup = BeautifulSoup(response.text, "html.parser")

        # Locate all quote containers on the page
        quote_elements = soup.find_all("div", class_="quote")

        for element in quote_elements:
            # Extract text safely using inline checks to avoid AttributeError
            text_el = element.find("span", class_="text")
            author_el = element.find("small", class_="author")
            tags_els = element.find_all("a", class_="tag")

            text = text_el.text.strip() if text_el else None
            author = author_el.text.strip() if author_el else None
            tags = [tag.text.strip() for tag in tags_els] if tags_els else []

            # Append structured dictionary to our list
            all_quotes.append(
                {"quote": text, "author": author, "tags": ", ".join(tags)}
            )

        # Handle Pagination: Look for the 'Next' page button
        next_button = soup.find("li", class_="next")
        if next_button:
            next_link = next_button.find("a")
            current_page = next_link["href"] if next_link else None
        else:
            # No next button means we reached the final page
            current_page = None

        # Polite Scraping: Pause briefly between requests to not overwhelm the server
        time.sleep(1)

    print(f"Data collection complete! Total quotes scraped: {len(all_quotes)}")
    return all_quotes


# 2. Execute and Save Data
if __name__ == "__main__":
    # Scrape the data
    scraped_data = scrape_quotes()

    # Convert the list of dicts into a structured Pandas DataFrame
    df = pd.DataFrame(scraped_data)

    # Preview the data in the console
    print("\nData Preview:")
    print(df.head())

    # Export cleanly to CSV and JSON formats
    df.to_csv("scraped_quotes.csv", index=False, encoding="utf-8")
    df.to_json("scraped_quotes.json", orient="records", indent=4)
    print("\nFiles saved successfully as 'scraped_quotes.csv' and 'scraped_quotes.json'")

import re
class scraper:
    def __init__(self, response):
        self.response = response

    def scrape_csv(self):
        text = str(self.response)
        scraped_data = re.findall(r'```csv(.*?)```', text, re.DOTALL)
        return "\n".join(chunk.strip() for chunk in scraped_data).strip()

    def scrape_text(self):
        text = str(self.response)
        result = re.sub(r'```csv.*?```', '', text, flags=re.DOTALL)
        return result.strip()
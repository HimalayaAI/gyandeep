import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from response_scraper import scraper

mock_response = """
Here is the explanation for the derivative of x^2.
We use the limit definition:
f'(x) = lim_{h->0} (f(x+h) - f(x))/h

```csv
equation,reasoning,plot,x,y,connection,name
y = x^2,Original function,new,-2;-1;0;1;2,4;1;0;1;4,True,Parabola
y' = 2x,Derivative function,new,-2;-1;0;1;2,-4;-2;0;2;4,True,Derivative
```

I hope this helps!
"""

s = scraper(mock_response)
csv_data = s.scrape_csv()
text_data = s.scrape_text()

print("--- CSV DATA ---")
print(csv_data)
print("--- TEXT DATA ---")
print(text_data)

expected_csv = "equation,reasoning,plot,x,y,connection,name\ny = x^2,Original function,new,-2;-1;0;1;2,4;1;0;1;4,True,Parabola\ny' = 2x,Derivative function,new,-2;-1;0;1;2,-4;-2;0;2;4,True,Derivative"

if csv_data == expected_csv:
    print("\nCSV SCRAPING SUCCESSFUL")
else:
    print("\nCSV SCRAPING FAILED")

if "Parabola" in text_data: # It shouldn't be in text_data if it's in the csv block
    print("TEXT SCRAPING potentially FAILED (CSV found in text)")
else:
    print("TEXT SCRAPING SUCCESSFUL (CSV removed)")

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def parse_dish_name_and_servings(h2_tag):
    """Lấy dish_name và servings từ h2 tag"""
    dish_name = h2_tag.find(string=True, recursive=False)
    if dish_name:
        dish_name = dish_name.strip()
        dish_name = re.sub(r'^Nguyên\s+liệu\s+làm\s+', '', dish_name, flags=re.IGNORECASE).strip()
    else:
        dish_name = ""
    
    servings = None
    small = h2_tag.find('small')
    if small:
        match = re.search(r'(\d+)', small.text)
        if match:
            servings = int(match.group(1))
    
    return dish_name, servings

def parse_quantity_unit(amount_text):
    """Tách quantity và unit từ text"""
    if not amount_text:
        return None, None
    
    amount_text = amount_text.strip()
    
    # "500 gram" -> 500, "gram"
    match = re.match(r'^(\d+(?:[.,]\d+)?)\s+(.+)$', amount_text)
    if match:
        return float(match.group(1).replace(',', '.')), match.group(2).strip()
    
    # "1/2 kg" -> 0.5, "kg"
    match = re.match(r'^(\d+)/(\d+)\s+(.+)$', amount_text)
    if match:
        return float(match.group(1)) / float(match.group(2)), match.group(3).strip()
    
    # "500" -> 500, None
    match = re.match(r'^(\d+(?:[.,]\d+)?)$', amount_text)
    if match:
        return float(match.group(1).replace(',', '.')), None
    
    return None, amount_text

def crawl_recipe(url):
    """Crawl 1 recipe"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        staple_div = soup.find('div', class_='staple')
        if not staple_div:
            return None
        
        h2 = staple_div.find('h2')
        dish_name, servings = parse_dish_name_and_servings(h2) if h2 else ("", None)
        
        ingredients = []
        for span in staple_div.find_all('span'):
            # Lấy tên nguyên liệu
            ingredient_name = ' '.join([t.strip() for t in span.find_all(string=True, recursive=False)]).strip()
            
            # Lấy quantity từ small
            small = span.find('small')
            amount_text = small.text.strip() if small else ""
            quantity, unit = parse_quantity_unit(amount_text)
            
            # Kiểm tra nếu là "Gia vị thông dụng"
            em_tag = span.find('em')
            if em_tag and 'gia vị' in ingredient_name.lower():
                # Tách các gia vị từ thẻ em
                spices_text = re.sub(r'^\(|\)$', '', em_tag.text.strip())
                for spice in re.split(r'[,/]', spices_text):
                    spice = spice.strip().capitalize()
                    if spice:
                        ingredients.append({'name': spice, 'quantity': quantity, 'unit': unit})
            elif ingredient_name:
                ingredients.append({'name': ingredient_name, 'quantity': quantity, 'unit': unit})
        
        return {
            'dish_name': dish_name,
            'url': url,
            'servings': servings,
            'ingredients': ingredients
        }
    
    except Exception as e:
        print(f"  Lỗi: {e}")
        return None

def main():
    df = pd.read_csv('data/recipe_urls.csv')
    
    recipes = []
    for i in range(len(df)):
        row = df.iloc[i]
        url = row['url']
        
        print(f"\n[{i+1}/{len(df)}] {url.split('/')[-1][:60]}")
        
        recipe = crawl_recipe(url)
        if recipe:
            recipe['category'] = row['category']
            recipes.append(recipe)
        
        time.sleep(0.5)
    
    with open('data/recipes_detail.json', 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Đã crawl {len(recipes)} món")
    print("=" * 60)

if __name__ == "__main__":
    main()
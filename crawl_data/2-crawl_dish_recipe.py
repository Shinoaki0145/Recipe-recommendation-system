import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# def parse_cooking_info(soup):
#     """Lấy thời gian chuẩn bị, chế biến và độ khó"""
#     prep_time, cook_time, difficulty = None, None, None
#     ready_ul = soup.find('ul', class_='ready')
#     if ready_ul:
#         for li in ready_ul.find_all('li'):
#             h2_text = li.find('h2').text.strip() if li.find('h2') else ""
#             span_text = li.find('span').text.strip() if li.find('span') else ""
            
#             if "Chuẩn bị" in h2_text:
#                 prep_time = span_text
#             elif "Chế biến" in h2_text:
#                 cook_time = span_text
#             elif "Độ khó" in h2_text:
#                 difficulty = span_text
#     return prep_time, cook_time, difficulty

def parse_cooking_info(soup):
    """Lấy thời gian chuẩn bị, chế biến và độ khó"""
    prep_time, cook_time, difficulty = None, None, None
    ready_ul = soup.find('ul', class_='ready')
    times_values = []

    if ready_ul:
        for li in ready_ul.find_all('li'):
            h2 = li.find('h2')
            h2_text = h2.text.strip() if h2 else ""
            label_lower = h2_text.lower()
            span = li.find('span')
            span_text = span.text.strip() if span else ""

            if "độ khó" in label_lower:
                difficulty = span_text
                continue

            if span_text:
                times_values.append(span_text)

            if "chuẩn bị" in label_lower:
                prep_time = span_text
            elif "chế biến" in label_lower or "thực hiện" in label_lower:
                cook_time = span_text

        # Trường hợp chỉ có 1 thời gian (không phân loại rõ), ưu tiên gán vào prep_time
        if prep_time is None and cook_time is None and times_values:
            ready_text = ready_ul.get_text(" ", strip=True).lower()
            has_prep = "chuẩn bị" in ready_text
            has_cook = "chế biến" in ready_text or "thực hiện" in ready_text

            if has_prep and not has_cook:
                prep_time = times_values[0]
            elif has_cook and not has_prep:
                cook_time = times_values[0]
            else:
                prep_time = times_values[0]

        # Nếu vẫn chưa bắt được prep_time nhưng cook_time có và chỉ có 1 thời gian, chuyển sang prep_time
        if prep_time is None and cook_time is not None and len(times_values) == 1:
            prep_time, cook_time = cook_time, None

    return prep_time, cook_time, difficulty


def parse_instructions(soup):
    """Lấy các bước thực hiện (Cách làm)"""
    instructions = []
    method_div = soup.find('div', class_='method')
    if method_div:
        # Tìm các li có id bắt đầu bằng 'step'
        for li in method_div.find_all('li', id=re.compile(r'^step')):
            step_title = li.find('h3').text.strip() if li.find('h3') else ""
            p_tags = li.find_all('p')
            step_content = " ".join([p.text.strip() for p in p_tags if p.text.strip()])
            
            if step_title or step_content:
                instructions.append({
                    'step_title': step_title,
                    'content': step_content
                })
    return instructions

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
        servings = small.get_text(" ", strip=True)

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

def parse_views(soup):
    """Lấy số lượt xem, chỉ trả về số"""
    count_div = soup.find('div', class_='count-view')
    if not count_div:
        return None

    text = count_div.get_text(" ", strip=True)
    match = re.search(r'(\d[\d\.]*)', text)
    if not match:
        return None

    digits_only = match.group(1).replace('.', '')
    try:
        return int(digits_only)
    except ValueError:
        return None

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
        prep_time, cook_time, difficulty = parse_cooking_info(soup)
        views = parse_views(soup)
        instructions = parse_instructions(soup)
        
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
            'prep_time': prep_time,
            'cook_time': cook_time,
            'difficulty': difficulty,
            'views': views,
            'ingredients': ingredients,
            'instructions': instructions
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
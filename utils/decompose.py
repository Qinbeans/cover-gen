import json
from bs4 import BeautifulSoup, NavigableString

def get_text(element) -> str:
    """
    Get and concatenate only the strings from inside an element
    @param element: the element to get the text from
    @return: the text from the element
    """
    if isinstance(element, NavigableString):
        return element.text.replace('\xa0', ' ').replace('\u2019',"'")
    if isinstance(element, str):
        return element.replace('\xa0', ' ').replace('\u2019',"'")
    text = ""
    for child in element.children:
        text += get_text(child)
    return text

def find_ul_tags(element, tags: list = None) -> list:
    """
    Recursively find ul tags and return them in a list
    @param element: the element to find ul tags in
    @param tags: the tags to find
    @return: a list of ul tags
    """
    if isinstance(element, NavigableString):
        return []

    ul_tags_with_siblings = []

    for child in element.children:
        if tags is not None:
            for tag in tags:
                if tag in child.text.lower():
                    if tag == "c++/c" or tag == "c/c++":
                        ul_tags_with_siblings.append(("Skills", "c"))
                        ul_tags_with_siblings.append(("Skills", "c++"))
                    else:
                        ul_tags_with_siblings.append(("Skills", tag))
        if isinstance(child, NavigableString):
            continue
        if child.name == 'ul':
            # loop through ul tag children and add them to a list called ul_list
            ul_list = []
            for ul_child in child.children:
                if isinstance(ul_child, NavigableString):
                    continue
                text = get_text(ul_child).replace('\xa0', ' ').replace('\u2019',"'").replace('\u201c', '"').replace('\u201d', '"')
                ul_list.append(text)
            # reduce the sibling to a single string
            ul_tags_with_siblings.append((get_text(child.previous_sibling), ul_list))

        # recursively call the function on the child
        ul_tags_with_siblings += find_ul_tags(child)

    return ul_tags_with_siblings

def decompose_job(text: str) -> str:
    """
    Decompose the HTML into a JSON file
    @param text: the HTML text
    @return: the JSON file
    """
    key_headers = ["Responsibilities:", "Qualifications:", "Requirements:", "Skills:"]
    soup = BeautifulSoup(text, 'html.parser')

    body = soup.find("body")

    # find the ul tags
    tags = find_ul_tags(body)

    data = {}

    for tag in tags:
        if tag[0] is None:
            continue
        # check if tag[0] contains any of the key headers
        for key_header in key_headers:
            if key_header in tag[0]:
                data[key_header.replace(":","")] = tag[1]
                break
    json_data = json.dumps(data, indent=4)

    return json_data

def decompose_resume(text: str) -> str:
    """
    Decompose the HTML into a JSON file
    @param text: the HTML text
    @return: the JSON file
    """
    skills_tags = ["docker", "rust", "java", "golang", "rust", "ghidra", "python", "mysql", "redis", "sqlite", "json", "pytorch","c/c++","c++/c", "orm"]
    soup = BeautifulSoup(text, 'html.parser')
    # go through the body and find tags that contain the key headers
    body = soup.find("body")
    # find the ul tags
    tags = find_ul_tags(body, skills_tags)

    data = {}

    for tag in tags:
        if tag[0] is None:
            continue
        # check if tag[0] contains any of the key headers
        if "http" in tag[0]:
            # this is probably a project
            if data.get("Projects") is None:
                data["Projects"] = []
            data["Projects"].append({
                "title": tag[0].replace('\u2013', '-').replace('\u2022', '*'),
                "description": tag[1]
            })
        elif tag[0] == "Skills":
            if data.get("Skills") is None:
                data["Skills"] = []
            if not isinstance(tag[1], str):
                continue
            # check if tag[1] is in Skills already
            if tag[1] not in data["Skills"]:
                data["Skills"].append(tag[1])
        else:
            if data.get("Experience/Activities") is None:
                data["Experience/Activities"] = []
            data["Experience/Activities"].append({
                "title": tag[0].replace('\u2013', '-').replace('\u2022', '*'),
                "description": tag[1]
            })
    json_data = json.dumps(data, indent=4)

    return json_data
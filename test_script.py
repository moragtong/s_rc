import urllib.request
url = "https://raw.githubusercontent.com/bevyengine/bevy/v0.15.0-rc.3/examples/shader/custom_vertex_attribute.rs" # close to 0.19/git?
try:
    response = urllib.request.urlopen(url)
    print(response.read().decode('utf-8')[:500])
except Exception as e:
    print(e)

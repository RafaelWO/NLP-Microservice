import sys
import requests

if len(sys.argv) <= 1:
    sys.exit(-1)

ip_addr = sys.argv[1]

print("Making requests to IP", ip_addr)
print("[Enter 'q' to quit]")
print("[Enter 'c' to continue generating with the previous output]")

try:
    usr_input = input("Text prompt:")
    prev_text = ""
    while usr_input != "q":
        if usr_input == "c":
            usr_input = prev_text
        req = requests.post(f"http://{ip_addr}/text-generation/generate",
                            json={'text': usr_input})
        response = req.json()
        print(response['generated'])
        prev_text = usr_input + response['generated']
        usr_input = input("Text prompt: ")
except Exception as ex:
    print(ex)

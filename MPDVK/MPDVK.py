#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, re, simplejson, sys, urllib, urllib2
from cookielib import CookieJar
from bs4 import BeautifulSoup
from mpd import MPDClient

if len(sys.argv) not in [4, 5]:
    print "Usage: %(argv)s <login> <password> <http://vk.com/...> <limit=100>" % {
        "argv"  : sys.argv[0]
    }
    sys.exit(1)

try:
    HOST = os.environ["MPD_HOST"]
except:
    HOST = "localhost"
try:
    PORT = int(os.environ["MPD_PORT"])
except:
    PORT = 6600

client = MPDClient()
client.connect(host=HOST, port=PORT)

cj = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

login_post_data = {}
login_page = BeautifulSoup(opener.open("http://oauth.vk.com/authorize?client_id=1&scope=8&redirect_uri=http://oauth.vk.com/blank.html&display=page&response_type=token").read())
for input in login_page.find_all("input"):
    if input.get("name"):
        login_post_data[input.get("name")] = input.get("value")
login_post_data["email"] = sys.argv[1]
login_post_data["pass"] = sys.argv[2]
login_post_data["expire"] = "0"

authorize_response = opener.open(login_page.find("form").get("action"), urllib.urlencode(login_post_data))
if "oauth.vk.com/blank.html" in authorize_response.geturl():
    oauth_success_url = authorize_response.geturl()
else:
    authorize_page = BeautifulSoup(authorize_response.read())
    oauth_success_url = opener.open(authorize_page.find("form").get("action")).geturl()
access_token = re.search("access_token=([0-9a-f]+)&", oauth_success_url).group(1)

group_id = int(re.search("/g([0-9]+)/", urllib2.urlopen(urllib2.Request(sys.argv[3])).read()).group(1))

client.clear()

i = 0
try:
    limit = int(sys.argv[4])
except:
    limit = 100
audios = []
for post in simplejson.loads(urllib2.urlopen(urllib2.Request("https://api.vk.com/method/wall.get?owner_id=-%(group_id)d?filter=owner&count=100" % {
    "group_id"  : group_id,
})).read())["response"][1:]:
    for attachment in post["attachments"]:
        if attachment["type"] == "audio":
            if i >= limit:
                break            
            i += 1

            url = simplejson.loads(urllib2.urlopen(urllib2.Request("https://api.vk.com/method/audio.getById?audios=%(audios)s&access_token=%(access_token)s" % {
                "audios"        : str(attachment["audio"]["owner_id"]) + "_" + str(attachment["audio"]["aid"]),
                "access_token"  : access_token,
            })).read())["response"][0]["url"]
            client.add(url)

            if i == 1:
                client.play()

client.disconnect()

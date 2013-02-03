#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import urllib2
from lxml import etree
import cgi, pynotify
import time

import NotifyGmailConfig

gmail_auth_handler = urllib2.HTTPBasicAuthHandler()
gmail_auth_handler.add_password(realm='New mail feed', uri='https://mail.google.com/mail/feed/atom', user=NotifyGmailConfig.user, passwd=NotifyGmailConfig.passwd)
gmail_opener = urllib2.build_opener(gmail_auth_handler)
pynotify.init("gmail")
shown = []
while True:
    for entry in etree.XML(gmail_opener.open('https://mail.google.com/mail/feed/atom').read()).findall('{http://purl.org/atom/ns#}entry'):
        id = entry.find('{http://purl.org/atom/ns#}id').text
        if id not in shown:
            title = entry.find('{http://purl.org/atom/ns#}title').text
            summary = entry.find('{http://purl.org/atom/ns#}summary').text
            # author = entry.find('{http://purl.org/atom/ns#}author').find('{http://purl.org/atom/ns#}name')
            pynotify.Notification(cgi.escape(title if title else u"<нет>"), cgi.escape(summary if summary else u"<нет>"), "/usr/share/icons/Tango/128x128/apps/internet-mail.png").show()
            shown.append(id)
    time.sleep(300)

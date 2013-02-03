#!/usr/bin/env python2
import SocketServer

def ReceivedImMsg(account, name, message, conversation, flags):
    user = purple.PurpleConversationGetTitle(conversation)

    import cgi
    from BeautifulSoup import BeautifulSoup
    pynotify.Notification(cgi.escape(user), cgi.escape("\n".join(BeautifulSoup(message).findAll(text=True))), "/usr/share/icons/Tango/128x128/apps/internet-group-chat.png").show()

from dbus.mainloop.glib import DBusGMainLoop
DBusGMainLoop(set_as_default=True)

import dbus
bus = dbus.bus.BusConnection("tcp:host=192.168.0.3,port=55555")
bus.add_signal_receiver(ReceivedImMsg, dbus_interface="im.pidgin.purple.PurpleInterface", signal_name="ReceivedImMsg")
purple = bus.get_object("im.pidgin.purple.PurpleService", "/im/pidgin/purple/PurpleObject")

import pynotify
pynotify.init("pidgin")

import gobject
loop = gobject.MainLoop()
loop.run()

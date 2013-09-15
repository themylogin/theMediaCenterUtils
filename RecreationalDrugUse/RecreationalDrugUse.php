#!/usr/bin/env php
<?php
print "\033]0;4-HO-MET\007";
while ("" == ($windowId = trim(`wmctrl -l | grep "4-HO-MET" | cut -f 1 -d " "`)));
system("wmctrl -ir $windowId -e 1000,-1,-88,1962,1201");
// for(;;){for($i=0;$i<122;$i++)echo chr(rand(32,128));echo"\n";usleep(rand(10000,30000));}
// for(;;){for($i=0;$i<122;$i++)echo"\033[".rand(0,8).";".(rand(0,1)?rand(30,37):rand(40,47))."m".chr(rand(32,128));echo"\n";usleep(rand(10000,30000));}
// for(;;){for($i=0;$i<122;$i++)echo"\033[".rand(1,1).";".(rand(1,1)?rand(30,37):rand(40,47))."m".chr(rand(32,128));echo"\n";usleep(rand(10000,30000));}

class RecreationalDrugUse
{
    static $mode = 0;
    static function toggleMode() { self::$mode = (self::$mode + 1) % 3; }

    static $speed = 20000;
    static function increaseSpeed() { self::$speed = max(0, self::$speed - 5000); return ""; }
    static function decreaseSpeed() { self::$speed += 2500; return ""; }
}
RecreationalDrugUse::$mode = isset($argv[1]) ? intval($argv[1]) : 0;

$dbus = new Dbus(Dbus::BUS_SESSION, true);
$dbus->requestName("ru.thelogin.RecreationalDrugUse");
$dbus->registerObject("/ru/thelogin/RecreationalDrugUse", "ru.thelogin.RecreationalDrugUse", "RecreationalDrugUse");

while (true)
{
    echo "\033[0;0m";
    switch (RecreationalDrugUse::$mode)
    {
        case 0: for($i=0;$i<122;$i++)echo chr(rand(32,128)); break;
        case 1: for($i=0;$i<122;$i++)echo"\033[".rand(0,0).";".(rand(1,1)?rand(30,37):rand(40,47))."m".chr(rand(32,128)); break;
        case 2: for($i=0;$i<122;$i++)echo"\033[".rand(0,8).";".(rand(0,1)?rand(30,37):rand(40,47))."m".chr(rand(32,128)); break;
    }
    echo "\n";

    usleep(rand(RecreationalDrugUse::$speed * 0.5, RecreationalDrugUse::$speed * 1.5));
    $dbus->waitLoop(1);
}

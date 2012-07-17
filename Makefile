all: mpd-wallpaper-changer

mpd-wallpaper-changer: image.o lastfm.o main.o mpd.o
	nvcc main.o image.o lastfm.o mpd.o -lcurl -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -lglib-2.0 -lImlib2 -lmpdclient -lpthread -lX11 -I/usr/include/libxml2/ -lxml2 -o mpd-wallpaper-changer

image.o: image.cu
	nvcc -c image.cu

main.o: main.cu
	nvcc -c main.cu

lastfm.o: lastfm.c
	g++ -c -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -I/usr/include/libxml2/ lastfm.c

mpd.o: mpd.c
	g++ -c mpd.c

clean:
	rm -rf *.o mpd-wallpaper-changer

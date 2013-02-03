#ifndef _LASTFM_H
#define _LASTFM_H

#include "mpd.h"

// Данные потока, работающего с last.fm
typedef struct {
    // Входные данные
    mpd_thread_data_t* mpd; // Данные MPD
    char* directory;        // Директория, куда скачивать картинки
    int poll_interval;      // Интервал опроса mpd_thread_data_t
    int expire_interval;    // Интервал опроса last.fm
} lastfm_thread_data_t; 

void* lastfm_thread_body(void* _data);

#endif

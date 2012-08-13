#ifndef _MPD_H
#define _MPD_H

#include <mpd/client.h>
#include <mpd/status.h>
#include <mpd/entity.h>
#include <mpd/tag.h>

#include <pthread.h>

// Данные потока, работающего с MPD
typedef struct {
    // Входные данные
    int timeout;        // Таймаут соединения с MPD-сервером
    int poll_interval;  // Интервал опроса MPD-сервера
    // Выходные данные
    int is_playing;                 // 1, если в настоящий момент играет песня, у которой известен исполнитель, 0, если нет
    char now_playing_artist[1024];  // Если is_playing == 1, то здесь лежит исполнитель
    char now_playing_track[1024];   // Если is_playing == 1, то здесь лежит трек
    // Потокобезопасность
    pthread_mutex_t mutex;
} mpd_thread_data_t; 

void* mpd_thread_body(void* _data);

#endif

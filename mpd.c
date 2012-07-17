#include "mpd.h"

#include <string.h>
#include <pthread.h>
#include <unistd.h>

// Безопасно установить is_playing в структуре
void _mpd_thread_set_is_playing(mpd_thread_data_t* data, int is_playing)
{
    pthread_mutex_lock(&data->mutex);
    data->is_playing = is_playing;
    pthread_mutex_unlock(&data->mutex);
}

// Поток периодически опрашивает MPD-сервер (переподключаясь в случае ошибки) на предмет воспроизводимой песни и особенно названия исполнителя
void* mpd_thread_body(void* _data)
{
    mpd_thread_data_t* data = (mpd_thread_data_t*)_data;

    // Цикл переподключения
    while (1)
    {
        // Подключение
        struct mpd_connection* conn = mpd_connection_new(NULL, 0, data->timeout);
        if (mpd_connection_get_error(conn) != MPD_ERROR_SUCCESS)
        {
            goto finish;
        }

        // Цикл опроса
        while (1)
        {
            // Отправляем запрос
            mpd_command_list_begin(conn, true);
            mpd_send_status(conn);
            mpd_send_current_song(conn);
            mpd_command_list_end(conn);

            // Получаем информацию о состоянии MPD-сервера
            struct mpd_status* status;
            if ((status = mpd_recv_status(conn)) == NULL)
            {
                goto finish;
            }
            if (mpd_status_get_state(status) == MPD_STATE_PLAY)
            {
                // Что-то воспроизводим, узнаем что
                struct mpd_song* song;
                mpd_response_next(conn);
                if ((song = mpd_recv_song(conn)) != NULL)
                {
                    const char *value;
                    if ((value = mpd_song_get_tag(song, MPD_TAG_ARTIST, 0)) != NULL)
                    {
                        // Узнали имя исполнителя. Отлично
                        pthread_mutex_lock(&data->mutex);
                        data->is_playing = 1;
                        strcpy(data->now_playing_artist, value);
                        pthread_mutex_unlock(&data->mutex);
                    }
                    else
                    {
                        // Не получилось узнать имя исполнителя
                        _mpd_thread_set_is_playing(data, 0);
                    }

                    mpd_song_free(song);
                }
                else
                {
                    // Не получилось узнать что
                    _mpd_thread_set_is_playing(data, 0);
                }
            }
            else
            {
                // Ничего не воспроизводим
                _mpd_thread_set_is_playing(data, 0);
            }

            if (mpd_connection_get_error(conn) != MPD_ERROR_SUCCESS)
            {
                goto finish;
            }

            sleep(data->poll_interval);
        }

        // Сюда переходим в случае ошибки: отключаемся и потом подключимся снова
        finish:
        _mpd_thread_set_is_playing(data, 0);
        mpd_connection_free(conn);
    }
}

#include "lastfm.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>

#define __deprecated__(msg) __deprecated__
#include <glib.h>

#include <pthread.h>

#include <curl/curl.h>

#include <libxml/parser.h>
#include <libxml/tree.h>

// http://curl.haxx.se/libcurl/c/getinmemory.html
typedef struct {
    char* memory;
    size_t size;
} lastfm_thread_body_write_memory_callback_chunk_t;
static size_t lastfm_thread_body_write_memory_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
    size_t realsize = size * nmemb;
    lastfm_thread_body_write_memory_callback_chunk_t* mem = (lastfm_thread_body_write_memory_callback_chunk_t*)userp;
    
    mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);    
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}

// http://curl.haxx.se/libcurl/c/ftpget.html
typedef struct {
    const char* filename;
    FILE* stream;
} lastfm_download_file_write_memory_callback_t;
static size_t lastfm_download_file_write_memory_callback(void* buffer, size_t size, size_t nmemb, void* stream)
{
    lastfm_download_file_write_memory_callback_t* out = (lastfm_download_file_write_memory_callback_t*)stream;
    if (out && !out->stream)
    {
        out->stream = fopen(out->filename, "wb");
        if (!out->stream)
        {
            return 0;
        }
    }
    
    return fwrite(buffer, size, nmemb, out->stream);
}

// Получить XML'ку arist.getimages с last.fm
char* lastfm_artist_getimages(char* artist, int page)
{
    printf("lastfm_artist_getimages(\"%s\", %d)\n", artist, page);
    
    lastfm_thread_body_write_memory_callback_chunk_t lastfm_thread_body_write_memory_callback_chunk;
    lastfm_thread_body_write_memory_callback_chunk.memory = (char*)malloc(1);
    lastfm_thread_body_write_memory_callback_chunk.size = 0;

    CURL* curl = curl_easy_init();
    if (curl)
    {
        char* artist_escaped = curl_easy_escape(curl, artist, 0);
        char url[1024];
        sprintf(url, "http://ws.audioscrobbler.com/2.0/?method=artist.getimages&artist=%s&page=%d&autocorrect=1&api_key=b25b959554ed76058ac220b7b2e0a026", artist_escaped, page);
        curl_free(artist_escaped);
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, lastfm_thread_body_write_memory_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&lastfm_thread_body_write_memory_callback_chunk);
        
        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK)
        {
            curl_easy_cleanup(curl);
            
            return lastfm_thread_body_write_memory_callback_chunk.memory;
        }        
        curl_easy_cleanup(curl);
    }

    return NULL;
}

// Скачать файл
int lastfm_download_file(const char* url, const char* dst)
{
    lastfm_download_file_write_memory_callback_t stream;
    stream.filename = dst;
    stream.stream = NULL;
    
    CURL* curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, lastfm_download_file_write_memory_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stream);

        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK)
        {
            fclose(stream.stream);
            curl_easy_cleanup(curl);
            return 1;
        }
        
        curl_easy_cleanup(curl);
        if (stream.stream)
        {
            fclose(stream.stream);
        }
        unlink(dst);
    }
    
    return 0;
}

// Поток скачивает с last.fm картинки исполнителя, играющего в данный момент
void* lastfm_thread_body(void* _data)
{
    lastfm_thread_data_t* data = (lastfm_thread_data_t*)_data;

    curl_global_init(CURL_GLOBAL_ALL);
    LIBXML_TEST_VERSION

    char current_artist[1024];
    GHashTable* fully_downloaded_artists = g_hash_table_new(g_str_hash, g_str_equal);
    while (1)
    {
        pthread_mutex_lock(&data->mpd->mutex);
        if (data->mpd->is_playing)
        {
            strcpy(current_artist, data->mpd->now_playing_artist);
            pthread_mutex_unlock(&data->mpd->mutex);

            int pages = -1;
            if (time(NULL) - (int)g_hash_table_lookup(fully_downloaded_artists, current_artist) < data->expire_interval)
            {
                pages = 0; // Не скачивать этого исполнителя
            }
            for (int page = 1; pages < 0 || page <= pages; page++)
            {
                pages = 0;
                char* xml = lastfm_artist_getimages(current_artist, page);
                if (xml)
                {
                    xmlDocPtr doc = xmlReadMemory(xml, strlen(xml), "images.xml", NULL, 0);
                    if (doc)
                    {
                        xmlNode* lfm = xmlDocGetRootElement(doc);
                        if (!xmlStrcmp(lfm->name, (const xmlChar*)"lfm"))
                        {
                            for (xmlNode* images = lfm->children; images; images = images->next)
                            {                            
                                if (images->type == XML_ELEMENT_NODE && !xmlStrcmp(images->name, (const xmlChar*)"images"))
                                {
                                    xmlChar* totalPages = xmlGetProp(images, (const xmlChar*)"totalPages");
                                    pages = atoi((const char*)totalPages);

                                    for (xmlNode* image = images->children; image; image = image->next)
                                    {
                                        if (image->type == XML_ELEMENT_NODE && !xmlStrcmp(image->name, (const xmlChar*)"image"))
                                        {
                                            for (xmlNode* sizes = image->children; sizes; sizes = sizes->next)
                                            {
                                                if (sizes->type == XML_ELEMENT_NODE && !xmlStrcmp(sizes->name, (const xmlChar*)"sizes"))
                                                {
                                                    for (xmlNode* size = sizes->children; size; size = size->next)
                                                    {
                                                        if (size->type == XML_ELEMENT_NODE && !xmlStrcmp(size->name, (const xmlChar*)"size"))
                                                        {
                                                            if (!xmlStrcmp(xmlGetProp(size, (const xmlChar*)"name"), (const xmlChar*)"original"))
                                                            {
                                                                for (xmlNode* text = size->children; text; text = text->next)
                                                                {
                                                                    if (text->type == XML_TEXT_NODE)
                                                                    {
                                                                        // Обнаружен файл; скачаем, если его нет
                                                                        const char* image_url = (const char*)text->content;

                                                                        char* directory = (char*)malloc(strlen(data->directory) + 1 + strlen(current_artist) + 1);
                                                                        sprintf(directory, "%s/%s", data->directory, current_artist);
                                                                        struct stat st;
                                                                        if (stat(directory, &st) != 0)
                                                                        {
                                                                            mkdir(directory, 0755);
                                                                        }
                                                                        
                                                                        char* filename_part = strdup(image_url);
                                                                        for (int i = 0; filename_part[i] != '\0'; i++)
                                                                        {
                                                                            if (filename_part[i] == '/')
                                                                            {
                                                                                filename_part[i] = ' ';
                                                                            }
                                                                        }
                                                                        char* filename = (char*)malloc(strlen(directory) + 1 + strlen(filename_part) + 1);
                                                                        sprintf(filename, "%s/%s", directory, filename_part);

                                                                        if (stat(filename, &st) != 0)
                                                                        {
                                                                            lastfm_download_file(image_url, filename);
                                                                        }

                                                                        free(directory);
                                                                        free(filename_part);
                                                                        free(filename);

                                                                        // Проверим, вдруг музыка кончилась или сменилась на другую
                                                                        pthread_mutex_lock(&data->mpd->mutex);
                                                                        if (!data->mpd->is_playing || strcmp(current_artist, data->mpd->now_playing_artist))
                                                                        {
                                                                            pthread_mutex_unlock(&data->mpd->mutex);
                                                                            pages = 0;  // Чтобы не переходил к другой странице
                                                                            goto finishDoc;
                                                                        }
                                                                        pthread_mutex_unlock(&data->mpd->mutex);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        finishDoc:
                        xmlFreeDoc(doc);
                    }
                }
            }
            if (pages != 0)
            {
                // Полностью скачали исполнителя, больше не будем его скачивать
                g_hash_table_insert(fully_downloaded_artists, current_artist, (gpointer)time(NULL));
            }
        }
        else
        {
            pthread_mutex_unlock(&data->mpd->mutex);
        }

        sleep(data->poll_interval);
    }
}

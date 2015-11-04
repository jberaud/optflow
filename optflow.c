#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <getopt.h>
#include <linux/videodev2.h>

static struct option long_options[] = {
    {"input",     required_argument, 0,     'i' },
    {"width",     required_argument, 0,     'w' },
    {"meta",      no_argument,       0,     'm' },
    {"format",    required_argument, 0,     'f' },
    {"help",      no_argument,       0,     'h' },
    {0, 0, 0, 0 }
};

static const char usage[] = "usage:\n\toptflow -i input_file -w width -f"
                            "pix_format [-m]\n -m option indicates the"
                            "presence of metadatas\n\n";

struct __attribute__((packed)) metadata {
    uint32_t timestamp;
    float x;
    float y;
    float z;
};

struct metadata empty_meta = {0, 0, 0, 0};

uint32_t char2fmt(const char *strfmt)
{
    if (!strcmp(strfmt, "NV12")) {
        return V4L2_PIX_FMT_NV12;
    } else if (!strcmp(strfmt, "GREY")) {
        return V4L2_PIX_FMT_GREY;
    } else {
        return 0;
    }
}

#define SEARCH_SIZE 4 // maximum offset to search: 4 + 1/2 pixels
#define NUM_BLOCKS  6 // x & y number of tiles to check
const float _bottom_flow_feature_threshold = 30;
const float _bottom_flow_value_threshold = 5000;
const float _focal_length_millipx = 2.5 / (3.6 * 2.0 * 240 / 64);

/**
 * @brief Compute the average pixel gradient of all horizontal and vertical steps
 *
 * TODO compute_diff is not appropriate for low-light mode images
 *
 * @param image ...
 * @param offX x coordinate of upper left corner of 8x8 pattern in image
 * @param offY y coordinate of upper left corner of 8x8 pattern in image
 */
uint32_t compute_diff(uint8_t *image, uint16_t offX, uint16_t offY,
                      uint16_t row_size, uint8_t window_size)
{
    /* calculate position in image buffer */
    uint16_t off = (offY + 2) * row_size + (offX + 2); // we calc only the 4x4 pattern
    uint32_t acc = 0;
    unsigned int i;

    for (i = 0; i < window_size; i++) {
        /* accumulate differences between line1/2, 2/3, 3/4 for 4 pixels
         * starting at offset off
         */
        acc += abs(image[off + i] - image[off + i + row_size]);
        acc += abs(image[off + i + row_size] - image[off + i + 2 * row_size]);
        acc += abs(image[off + i + 2 * row_size] - image[off + i + 3 * row_size]);
        
        /* accumulate differences between col1/2, 2/3, 3/4 for 4 pixels starting
         * at off
         */
        acc += abs(image[off + row_size * i] - image[off + row_size * i + 1]);
        acc += abs(image[off + row_size * i + 1] - image[off + row_size * i + 2]);
        acc += abs(image[off + row_size * i + 2] - image[off + row_size * i + 3]);
    }

    return acc;
}

/**
 * @brief Compute SAD of two pixel windows.
 *
 * @param image1 ...
 * @param image2 ...
 * @param off1X x coordinate of upper left corner of pattern in image1
 * @param off1Y y coordinate of upper left corner of pattern in image1
 * @param off2X x coordinate of upper left corner of pattern in image2
 * @param off2Y y coordinate of upper left corner of pattern in image2
 */
static inline uint32_t compute_sad(uint8_t *image1, uint8_t *image2, uint16_t off1X, uint16_t off1Y, uint16_t off2X, uint16_t off2Y, uint16_t row_size, uint16_t window_size)
{
    /* calculate position in image buffer */
    uint16_t off1 = off1Y * row_size + off1X; // image1
    uint16_t off2 = off2Y * row_size + off2X; // image2
    unsigned int i,j;
    uint32_t acc = 0;
    
    for (i = 0; i < window_size; i++)
        for (j = 0; j < window_size; j++)
            acc += abs(image1[off1 + i + j*row_size] - image2[off2 + i + j*row_size]);

    return acc;
}

/**
 * @brief Compute SAD distances of subpixel shift of two pixel patterns.
 *
 * @param image1 ...
 * @param image2 ...
 * @param off1X x coordinate of upper left corner of pattern in image1
 * @param off1Y y coordinate of upper left corner of pattern in image1
 * @param off2X x coordinate of upper left corner of pattern in image2
 * @param off2Y y coordinate of upper left corner of pattern in image2
 * @param acc array to store SAD distances for shift in every direction
 */
static inline uint32_t compute_subpixel(uint8_t *image1, uint8_t *image2, uint16_t off1X, uint16_t off1Y, uint16_t off2X, uint16_t off2Y, uint32_t *acc, uint16_t row_size, uint16_t window_size)
{
    /* calculate position in image buffer */
    uint16_t off1 = off1Y * row_size + off1X; // image1
    uint16_t off2 = off2Y * row_size + off2X; // image2
    uint8_t sub[8];
    uint16_t i, j, k;

    memset(acc, 0, window_size * sizeof(uint32_t));

    for (i = 0; i < window_size; i++)
    {
        for (j = 0; j < window_size; j++) {
            /* the 8 s values are from following positions for each pixel (X):
             *  + - + - + - +
             *  +   5   7   +
             *  + - + 6 + - +
             *  +   4 X 0   +
             *  + - + 2 + - +
             *  +   3   1   +
             *  + - + - + - +
            */

            /* subpixel 0 is the mean value of base pixel and
             * the pixel on the right, subpixel 1 is the mean
             * value of base pixel, the pixel on the right,
             * the pixel down from it, and the pixel down on
             * the right. etc...
             */
            sub[0] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i + 1 + j*row_size])/2;

            sub[1] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i + 1 + j*row_size] +
                      image2[off2 + i + (j+1)*row_size] +
                      image2[off2 + i + 1 + (j+1)*row_size])/4;

            sub[2] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i + 1 + (j+1)*row_size])/2;

            sub[3] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i - 1 + j*row_size] +
                      image2[off2 + i - 1 + (j+1)*row_size] +
                      image2[off2 + i + (j+1)*row_size])/4;

            sub[4] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i - 1 + (j+1)*row_size])/2;

            sub[5] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i - 1 + j*row_size] +
                      image2[off2 + i - 1 + (j-1)*row_size] +
                      image2[off2 + i + (j-1)*row_size])/4;

            sub[6] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i + (j-1)*row_size])/2;

            sub[7] = (image2[off2 + i + j*row_size] +
                      image2[off2 + i + 1 + j*row_size] +
                      image2[off2 + i + (j-1)*row_size] +
                      image2[off2 + i + 1 + (j-1)*row_size])/4;
            
            for (k = 0; k < 8; k++)
                acc[k] += abs(image1[off1 + i + j*row_size] - sub[k]);
        }
    }

    return 0;
}

uint8_t compute_flow(uint32_t width, void *image1, void *image2, uint32_t delta_time,
        float x_rate, float y_rate, float *pixel_flow_x, float *pixel_flow_y)
{
        /* constants */
    const int16_t winmin = -SEARCH_SIZE;
    const int16_t winmax = SEARCH_SIZE;

    /* variables */
    /* pixLo is SEARCH_SIZE + 1 because if we need to evaluate
     * the subpixels up/left of the first pixel, the index
     * will be equal to pixLo - SEARCH_SIZE -1
     * idem if we need to evaluate the subpixels down/right
     * the index will be equal to pixHi + SEARCH_SIZE + 1
     * which needs to remain inferior to width - 1
     */
    uint16_t pixLo = SEARCH_SIZE + 1;
    uint16_t pixHi = width - 1 - (SEARCH_SIZE + 1);
    uint16_t pixStep = (pixHi - pixLo) / NUM_BLOCKS + 1;
    uint16_t i, j;
    uint32_t acc[2*SEARCH_SIZE]; // subpixels
    int8_t  dirsx[NUM_BLOCKS*NUM_BLOCKS]; // shift directions in x
    int8_t  dirsy[NUM_BLOCKS*NUM_BLOCKS]; // shift directions in y
    uint8_t  subdirs[NUM_BLOCKS*NUM_BLOCKS]; // shift directions of best subpixels
    float meanflowx = 0.0f;
    float meanflowy = 0.0f;
    uint16_t meancount = 0;
    float histflowx = 0.0f;
    float histflowy = 0.0f;

    /* iterate over all patterns
     */
    for (j = pixLo; j < pixHi; j += pixStep)
    {
        for (i = pixLo; i < pixHi; i += pixStep)
        {
            /* test pixel if it is suitable for flow tracking */
            uint32_t diff = compute_diff(image1, i, j, (uint16_t) width, SEARCH_SIZE);
            if (diff < _bottom_flow_feature_threshold)
            {
                continue;
            }

            uint32_t dist = 0xFFFFFFFF; // set initial distance to "infinity"
            int8_t sumx = 0;
            int8_t sumy = 0;
            int8_t ii, jj;

            for (jj = winmin; jj <= winmax; jj++)
            {
                for (ii = winmin; ii <= winmax; ii++)
                {
                    uint32_t temp_dist = compute_sad(image1, image2, i, j, i + ii, j + jj,
                            (uint16_t) width, 2 * SEARCH_SIZE);
                    if (temp_dist < dist)
                    {
                        sumx = ii;
                        sumy = jj;
                        dist = temp_dist;
                    }
                }
            }

            /* acceptance SAD distance threshhold */
            if (dist < _bottom_flow_value_threshold)
            {
                meanflowx += (float) sumx;
                meanflowy += (float) sumy;

                compute_subpixel(image1, image2, i, j, i + sumx, j + sumy, acc, (uint16_t) width,
                        2 * SEARCH_SIZE);
                uint32_t mindist = dist; // best SAD until now
                uint8_t mindir = 8; // direction 8 for no direction
                for(uint8_t k = 0; k < 2 * SEARCH_SIZE; k++)
                {
                    if (acc[k] < mindist)
                    {
                        // SAD becomes better in direction k
                        mindist = acc[k];
                        mindir = k;
                    }
                }
                dirsx[meancount] = sumx;
                dirsy[meancount] = sumy;
                subdirs[meancount] = mindir;
                meancount++;
            }
        }
    }

    /* evaluate flow calculation */
    if (meancount > NUM_BLOCKS*NUM_BLOCKS/2)
    {
        meanflowx /= meancount;
        meanflowy /= meancount;

        /* use average of accepted flow values */
        uint32_t meancount_x = 0;
        uint32_t meancount_y = 0;

        for (uint16_t h = 0; h < meancount; h++)
        {
            float subdirx = 0.0f;
            if (subdirs[h] == 0 || subdirs[h] == 1 || subdirs[h] == 7) subdirx = 0.5f;
            if (subdirs[h] == 3 || subdirs[h] == 4 || subdirs[h] == 5) subdirx = -0.5f;
            histflowx += (float)dirsx[h] + subdirx;
            meancount_x++;

            float subdiry = 0.0f;
            if (subdirs[h] == 5 || subdirs[h] == 6 || subdirs[h] == 7) subdiry = -0.5f;
            if (subdirs[h] == 1 || subdirs[h] == 2 || subdirs[h] == 3) subdiry = 0.5f;
            histflowy += (float)dirsy[h] + subdiry;
            meancount_y++;
        }

        histflowx /= meancount_x;
        histflowy /= meancount_y;

        *pixel_flow_x = histflowx;
        *pixel_flow_y = histflowy;
    }
    else
    {
        *pixel_flow_x = 0.0f;
        *pixel_flow_y = 0.0f;
        return 0;
    }

    /* calc quality */
    uint8_t qual = (uint8_t)(meancount * 255 / (NUM_BLOCKS*NUM_BLOCKS));

    return qual;
}

int main(int argc, char **argv)
{
    int fd = -1;
    int c;
    uint32_t fmt;
    const char *in_path = NULL;
    uint8_t *frame = NULL;
    uint8_t *last_frame = NULL;
    uint32_t frame_size;
    uint32_t width;
    struct metadata *meta = &empty_meta;
    struct metadata *last_meta;
    float flow_x, flow_y;
    uint8_t qual;
    bool have_meta = false;

    while (1) {

        c = getopt_long(argc, argv, "i:w:mf:h", long_options, NULL);
        if (c == -1)
            break;

        switch (c) {
        case 'i':
            in_path = optarg;
            break;
        case 'w':
            width = strtoul(optarg, NULL, 10);
            break;
        case 'm':
            have_meta = true;
            break;
        case 'f':
            fmt = char2fmt(optarg);
            break;
        case 'h':
            printf(usage);
            goto end;
        case '?':
            printf(usage);
            goto end;
        default:
            fprintf(stderr, "bad option parsed by getopt : %d\n", c);
            goto end;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "wrong option in arguments: ");
        while (optind < argc)
            printf("%s", argv[optind++]);
        printf("\n");
    }

    fd = open(in_path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "can't open file %s\n", in_path);
        goto end;
    }
    if (width == 0 || width > 240) {
        fprintf(stderr, "width must be between 0 and 240\n");
        goto end;
    }

    if ((fmt != V4L2_PIX_FMT_NV12) && (fmt != V4L2_PIX_FMT_GREY)) {
        fprintf(stderr, "bad format %u\n", fmt);
        goto end;
    }

    frame_size = width * width;
    if (fmt == V4L2_PIX_FMT_NV12) {
        frame_size = frame_size * 3 / 2;
    }
    if (have_meta) {
        frame_size += sizeof(struct metadata);
    }

    while (1) {
        size_t bytes_read;
        frame = (uint8_t *) malloc(frame_size);

        if (frame == NULL) {
            fprintf(stderr, "failed to allocate %d bytes for frame\n", frame_size);
            break;
        }

        bytes_read = read(fd, frame, frame_size);

        if (have_meta)
            meta = (struct metadata *) (frame + frame_size - sizeof(struct metadata));
        
        if (bytes_read < frame_size) {
            break;
        } else if (last_frame == NULL) {
            goto loop;
        } else {
            qual = compute_flow(width, last_frame, frame, meta->timestamp - 
                    last_meta->timestamp, meta->x, meta->y, &flow_x, &flow_y);
            flow_x = flow_x / _focal_length_millipx /
                ((float)(meta->timestamp - last_meta->timestamp) / 1000.0f);
            flow_y = flow_y / _focal_length_millipx /
                ((float)(meta->timestamp - last_meta->timestamp) / 1000.0f);
            printf("flowx = %f, GyrX = %f, flowy = %f, GyrY = %f, qual %u\n",
                    flow_x, meta->x, flow_y, meta->y, qual);
        }

        free(last_frame);
loop:
        last_frame = frame;
        last_meta = meta;
    }

end:
    if (last_frame != 0)
        free(last_frame);
    if (frame != 0)
        free(frame);
    if (fd != -1)
        close(fd);

    return 0;
}

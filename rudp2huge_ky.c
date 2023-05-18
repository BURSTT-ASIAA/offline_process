/*
$ gcc -o rudp2shared rudp2shared.c -lrt -lgsl -lgslcblas -lm
to check hex file
$ hexdump -C /dev/shm/fpga1_shm | head -n 5
$ xxd /dev/shm/fpga1_shm | less
*/
#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>       /* For O_* constants */
#include <sys/mman.h>
#include <asm/mman.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <signal.h>
#include <syslog.h>
#include <mqueue.h>
#include <sched.h>

// NIC
#define MYPORT 60000    // the port users will be connecting to
#define MAXBUFLEN 9000
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
#define ADDR (void *) (0x0UL)   //starting address of the page

// message queue: /dev/mqueue/burstt
#define SERVER_QUEUE_NAME   "/burstt"
#define QUEUE_PERMISSIONS 0660
#define MAX_MESSAGES 20
#define MAX_MSG_SIZE 256
#define MSG_BUFFER_SIZE MAX_MSG_SIZE + 10

unsigned long packet_size = 8256;
unsigned long packet_number = 1000000;
unsigned int block = 12;
unsigned long block_size;
unsigned long long z, ivalue;
unsigned int opened=0;

int 	status=0;
bool 	loop=true;
	typedef struct {
                int command;
        } r_sock;

struct shmseg {
                int command;
        };
struct shmseg *shmp;

void setBit(unsigned char *bufstart, unsigned int index) {
        unsigned int p=0;
        unsigned int bit;
        unsigned char *buf;
        p = index/8;
        bit = index % 8;
        buf = bufstart +  p;
        *buf = *buf | 1 << bit;
}

int set_low_latfency() {
	uint32_t lat = 0;
	int fd = open("/dev/cpu_dma_latency", O_RDWR);
	if (fd == -1) {
		fprintf(stderr, "Failed to open cpu_dma_latency: error %s", strerror(errno));
		return fd;
	}
	write(fd, &lat, sizeof(lat));
	return fd;
}


void sigroutine (int sig) {
	printf("\nGot signal %d\n", sig);
	printf("Stop receiving packets shortly...\n\n");
	syslog(LOG_INFO, "Got signal %d", sig);
	syslog(LOG_INFO, "Stop receiving packets shortly");
	status = -1;
	loop = false;
	shmp->command = -1;
}

void watchdog()
{
	//loop = false;
	status = -1;
	shmp->command = -1;
	syslog(LOG_INFO, "Receiving Packets Timeout");
}

void low_data_rate()
{
        loop = false;
        status = -1;
	shmp->command = -1;
        syslog(LOG_INFO, "Process terminated due to low data rate");

}

void mq_error()
{       
        loop = false;
        status = -1;
	shmp->command = -1;
        syslog(LOG_INFO, "Message Queue Error");

}

int main(int argc, char *argv[]) 
{
/*
        struct shmseg {
                int command;
        };
	struct shmseg *shmp;
*/
	struct shmseg *shmp_timer;
	int shmid, shmid_timer;
	unsigned int port=60000;
        int sockfd;
	//struct timespec start, end;
	clock_t start, end, wdtimer1, wdtimer2;
        unsigned int i, j, first=1;
        struct sockaddr_in server_addr;         // server address information
        struct sockaddr_in fpga_addr;   	// fpga's address information
        int addr_len, numbytes;
	char *addr;
	char *optval;
        //unsigned char buf[MAXBUFLEN];		// each byte contains 4 bit im + 4 bit real
	unsigned int affinity=0, core=1;	// CPU core for affinity
	unsigned long long index=0;
	//char *shmfile;
	char shmfile[50];
	char datafile[50];
	unsigned int bindex;
	block_size = packet_number * packet_size;
	unsigned int bitmap_size = (block * packet_number / 8);
  	time_t t = time(NULL);
  	struct tm file_tm;
        mqd_t qd_server;   // queue descriptors

        unsigned int fpga = 0;	// fpga: 0, 1, 2, 3
        typedef struct {
                int fpga;
                int index;
        } queue_s;

        queue_s mqueue;


	// define signal
	signal(SIGINT, sigroutine);
	signal(SIGALRM, watchdog);	// set packet receiving timeout
	signal(SIGUSR1, low_data_rate); // set low data rate alarm
	signal(SIGUSR2, mq_error); // message queue error

        for(i=1;i<argc; i++) {
                if(strcmp(argv[i], "-i")==0) optval=argv[i+1];
                if(strcmp(argv[i], "-ps")==0) packet_size=atol(argv[i+1]);
                if(strcmp(argv[i], "-pn")==0) packet_number=atol(argv[i+1]); 
                if(strcmp(argv[i], "-cpu")==0) {core=atoi(argv[i+1]); affinity=1;}
		if(strcmp(argv[i], "-port")==0) port=atoi(argv[i+1]);
		if(strcmp(argv[i], "-block")==0) block=atoi(argv[i+1]);
		if(strcmp(argv[i], "-fpga")==0) fpga=atoi(argv[i+1]);   // use 0,1,2,....
        }

        unsigned char *buf = (unsigned char *) malloc(packet_size*sizeof(unsigned char));         // each byte contains 4 bit im + 4 bit real
        unsigned char *bitmap = (unsigned char *) malloc(block*packet_number*sizeof(unsigned char));         // each byte contains 4 bit im + 4 bit real


	int ret = set_low_latfency();


	// prepare for shared memory and mmap
	// packet filename
	size_t filesize = (block*packet_size*packet_number+bitmap_size)*sizeof(unsigned char);
	sprintf(shmfile, "/mnt/%s/fpga.bin", optval);
	int fd = open(shmfile, O_CREAT | O_RDWR, 0666);
	ftruncate(fd, filesize);

	int datafd=0;
	unsigned char *dest;

	// to attach the bitmap file (which contains which packets are received) to the end of packets received
        unsigned char *p = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB | MAP_HUGE_1GB, fd, 0);
	//unsigned char *dest = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, datafd, 0);
	unsigned char *pstart = p;
	unsigned char *pend = p + block * packet_size * packet_number * sizeof(unsigned char);
	unsigned char *bstart = buf;
	unsigned char *bitmapstart = bitmap;

        if (p == MAP_FAILED)    {
                printf("Mapping Failed\n");
                return 1;
        }

	memset(p, 0, block*packet_number*packet_size*sizeof(unsigned char));
	memset(bitmap, 0, bitmap_size*sizeof(unsigned char));
        // end of shared memory
	
        // define message queue
        struct mq_attr attr;
        attr.mq_flags = 0;
        //attr.mq_flags = O_NONBLOCK;
        attr.mq_maxmsg = MAX_MESSAGES;
        attr.mq_msgsize = MAX_MSG_SIZE;
        attr.mq_curmsgs = 0;
/*
        if ((qd_server = mq_open (SERVER_QUEUE_NAME, O_WRONLY)) == -1) {
        //if ((qd_server = mq_open (SERVER_QUEUE_NAME, O_WRONLY | O_CREAT)) == -1) {
                perror ("Client: mq_open (server)");
                exit (1);
        }
*/
        // end of message queue setup





	// setup CPU affinity
	if (affinity) {
        	cpu_set_t  mask;
        	CPU_ZERO (&mask);
		CPU_SET(core, &mask);
		int result = sched_setaffinity(0, sizeof(mask), &mask);
		if (result<0) {
                	perror("CPU affinity");
                	exit(1);
		}
	}
	// end of CPU affinity
	

	// set up shared memory
        //if ((shmid = shmget(0x1234, sizeof(struct shmseg), IPC_CREAT | 0666)) < 0) {
        if ((shmid = shmget(0x1234, sizeof(struct shmseg), 0666)) < 0) {
                perror("shmget");
                exit(1);
        }

        if ((shmp = shmat(shmid, NULL, 0)) == (void*) - 1) {
                perror("shmat");
                exit(1);
        }


	// socket for FPGA packets
	if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
                perror("socket");
                exit(1);
        }

	// set up nonblock
	fcntl(sockfd, F_SETFL, O_NONBLOCK);

        // assign NIC
        if ((setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, optval, strlen(optval)))<0) {
                perror("setsockopt bind device");
                exit(1);
        }

        //int rcvsize = 805306368;
        int rcvsize = 512*1024*1024;    // no receiver buffer error on kolong 
        if ((setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *)&rcvsize, (int)sizeof(rcvsize))) < 0)  {
                perror("rcvsize");
                exit(1);
        }
	
	int disable = 1;
	if ((setsockopt(sockfd, SOL_SOCKET, SO_NO_CHECK, (void*)&disable, sizeof(disable))) < 0) {
    		perror("setsockopt failed");
	}	

	// set receive timeout to 2 sec
	struct timeval read_timeout;
	read_timeout.tv_sec = 2;
	read_timeout.tv_usec = 0;
	setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof read_timeout);

        server_addr.sin_family = AF_INET;         // host byte order
        server_addr.sin_port = htons(MYPORT);     // short, network byte order
        server_addr.sin_addr.s_addr = INADDR_ANY; // automatically fill with my IP
	memset(&(server_addr.sin_zero), '\0', 8); // zero the rest of the struct

        if ((bind(sockfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr))) == -1) {
                        perror("bind");
                exit(1);
        }

        addr_len = sizeof(struct sockaddr);

	uint64_t t0;
	shmp->command = status;
	unsigned long int index_counter=1;	// index_counter increases 1 for every packet_number;
	unsigned int block_packet_number = block*packet_number;
	index_counter=1;
	float dr=0;
	unsigned int low_rate=0, dr_low_limit=6.2;
	printf("Waiting for sendsocket command 0:stop 1:start 2:stop/save -1:exit\n");
	//index_counter++;
	while (status>=0) {
		alarm(0);	// so when status = 0 is set, the watchdog will timeout
		status = shmp->command;
		usleep(1);

		if (status==1) {

			// clear bitmap buffer
			memset(bitmap, 0, bitmap_size*sizeof(unsigned char));

                        if ((numbytes=recvfrom(sockfd, buf, packet_size , 0, NULL, NULL)) == packet_size) {
                        	// index counter has 40 bit
                                // 2^40 / 10^6 * 1.27 / 86400 ~ 16 days
                                memcpy(&t0, buf, 5*sizeof(unsigned char));
                                index_counter = t0/packet_number;
                        }

			for (bindex=0; bindex < block; bindex++) {
				alarm(2);	// if the for loop does not come to here within 2 seconds, the alarm will be set off, or use ualarm()
				z = 0;
				loop = true;
                        	start = clock();
				
				p = pstart + bindex * block_size * sizeof(unsigned char);
			
				while(loop) 
					{	
						if ((numbytes=recvfrom(sockfd, buf, packet_size , 0, NULL, NULL)) == packet_size) {
							// index counter has 40 bit
							// 2^40 / 10^6 * 1.27 / 86400 ~ 16 days
							memcpy(&t0, buf, 5*sizeof(unsigned char));
							if (t0>0 && t0<=2) {printf("reset\n"); index_counter=0; alarm(2);} // may increase to alarm(3) for testing

							if (t0 > (unsigned long long int) ((index_counter+1)*packet_number)) loop=false;
							p = pstart + (t0%(block*packet_number))*packet_size * sizeof(unsigned char);
							memcpy(p, buf, packet_size);
							setBit(bitmap, t0%(block*packet_number));
							z++;
						}

					}
				index_counter++;
				end = clock();	

				dr = (packet_size*z/1e9)/(((double) (end-start))/CLOCKS_PER_SEC);
        			printf("%d cpu time used to read %ld UDP packets to hugepage: %.3f sec, ", bindex, z, ((double) (end-start))/CLOCKS_PER_SEC);
        			printf("data speed: %.3f GB/sec\n", dr);
			
			/*	
				if ( dr < dr_low_limit ){
					syslog(LOG_ERR, "NIC %s has low data rate %.3f GB/sec", optval,  dr);
					low_rate++;
					if (low_rate>3) {
						raise(SIGUSR1);
						shmp->command=-1;
					}
				}
			*/
			// need to update the bitmap mask to
			// p_mask = p_end + 125000 * bindex;	// 1000000/8 = 125000;
      				memcpy(pend, bitmap, bitmap_size); 	// if the computer is fast enough, then memcpy them all

                                // send out message queue
                                mqueue.fpga=fpga;
                                mqueue.index=bindex;
		/*	
                                if (mq_send (qd_server, (const char *) &mqueue, sizeof(mqueue), 0) == -1) {
					raise(SIGUSR2);
					shmp->command = -1;
                                        //perror ("Client: Not able to send message to server");
                                        //continue;
                                }
		*/	
                                // message queue sent

			} // bindex
		} // if status

		if (status==2) {

			alarm(0);
			time(&t);
			file_tm = *localtime(&t);
			usleep(1);
			sprintf(datafile, "/disk1/221230_11F_sun/%s.%02d%02d%02d%02d%02d.fpga.bin", optval, file_tm.tm_mon + 1, file_tm.tm_mday, file_tm.tm_hour, file_tm.tm_min, file_tm.tm_sec);
			//sprintf(datafile, "/mnt/%s.fpga.bin", optval, file_tm.tm_mon + 1, file_tm.tm_mday, file_tm.tm_hour, file_tm.tm_min, file_tm.tm_sec);
        		datafd = open(datafile, O_CREAT | O_RDWR | O_LARGEFILE, 0666);
        		ftruncate(datafd, filesize);
			dest = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, datafd, 0);
			opened = 1;

			printf("Stop receiving packets\n");
        		printf("Writing data %s\n", datafile);
        		p = pstart;
        		memcpy(dest, p, filesize);
			close(datafd);
			//shmp->command = 0;
			memset(p, 0, block*packet_number*packet_size*sizeof(unsigned char));
			shmp->command = 1;
			//printf("Waiting for sendsocket command 0:stop 1:start 2:stop/save -1:exit\n");
		}


	} // while status

        if (shmdt(shmp) == -1) {
                fprintf(stderr, "shmdt failed\n");
                exit(EXIT_FAILURE);
        }

	munmap(p, filesize);
    	if(opened) munmap(dest, filesize);

	close(sockfd);
	
	close(fd);
	//if(datafd>0) {
		//fflush(datafile);
		//close(datafd);
	//}

	free(buf);
	free(bitmap);
        return 0;
}


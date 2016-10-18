#include<iostream>
#include<opencv2/opencv.hpp>   
#include <cstring> 
#include <cmath> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>
#include<fstream>
using namespace std;

const char* cascade_face_name = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
CvHaarClassifierCascade* cascade_face = (CvHaarClassifierCascade*)cvLoad( cascade_face_name, 0, 0, 0 );

void HistogramMatch(IplImage*image1,IplImage*image2);
int getHopCount(uchar i);  
void lbp59table(uchar* table);
void LBP(IplImage* src, IplImage* dst);
void CutImage(IplImage*image1,IplImage*image2);

double persent;
string img;
string end1;

int main()
{
	
	
	      cout<<"计算两张人脸的相似度： "<<endl;
	      cout<<"请输入第一张静态图片的地址 ： ";
	      char s1[100];  cin>>s1;
	      char* filename1 = s1;

		  /*cout<<"请输入第二张静态图片的地址 ： ";
	      char s2[100];  cin>>s2;
	      char* filename2 = s2;*/

		  vector<string> img_path;//输入文件名变量      
		  string buf;    
		  ifstream svm_data( "D:/num.txt" );//样本图片的路径都写在这个txt文件中   
		  while( svm_data )//将训练样本文件依次读取进来    
		 {    
			if( getline( svm_data, buf ) )    
			{   
                img_path.push_back( buf );//图像路径    
            }    
         }    
      
 

		 for(int x=0;x< img_path.size();x++)
		 {
		   img=img_path[x];
		  // cvNamedWindow( "result1", CV_WINDOW_AUTOSIZE );
           IplImage* image1 = cvLoadImage( filename1, 1 );

		   //cvNamedWindow( "result2",CV_WINDOW_AUTOSIZE );
           IplImage* image2 = cvLoadImage( img_path[x].c_str(), 1 );
        
           if( image1&&image2 ) 
          { 
			  HistogramMatch(image1,image2);
              cvWaitKey(0); 
              cvReleaseImage( &image1 );  
			  cvReleaseImage( &image2 ); 
           }
		 }
          // cvDestroyWindow("result1");  
		  // cvDestroyWindow("result2");  
		  // cvDestroyWindow("face1");  
		  // cvDestroyWindow("face2"); 
		 cout<<"The best:  ";
		 cout<<persent<<endl<<end1;
		 system("pause");
	
}

//相似度计算函数
void HistogramMatch(IplImage*image1,IplImage*image2)
{

	double scale0 = 500.0/(image1 ->width);
	double scale1 = 500.0/(image2 ->width);

    static CvScalar colors[] = { 
        {{0,0,255}},{{0,128,255}},{{0,255,255}},{{0,255,200}}, 
        {{255,128,0}},{{255,255,0}},{{255,200,0}},{{255,200,255}} 
    };//Just some pretty colors to draw with

	/*CvHaarClassifierCascade* cascade_face = 0;
    const char* cascade_face_name = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
    cascade_face = (CvHaarClassifierCascade*)cvLoad( cascade_face_name, 0, 0, 0 );*/
	   


    //Image Preparation (the first image)
	IplImage* img1 =cvCreateImage(cvSize(cvRound(image1->width * scale0),cvRound(image1->height * scale0)),image1->depth,image1->nChannels);
	cvResize(image1, img1, CV_INTER_LINEAR);

    IplImage* gray1 = cvCreateImage(cvSize(img1->width,img1->height),8,1); 
	cvCvtColor(img1,gray1, CV_BGR2GRAY);  //Gray
	
	IplImage* smooth1 = cvCreateImage(cvSize(img1->width,img1->height),8,1); 
	cvSmooth(gray1,smooth1,CV_MEDIAN);//smooth

    cvEqualizeHist(smooth1,smooth1); //Histogram equalization

    //Detect objects if any 
   
	   CvMemStorage* storage_face1 = 0;
       storage_face1 = cvCreateMemStorage(0) ;
       cvClearMemStorage(storage_face1); 

    double t1 = (double)cvGetTickCount(); 

    CvSeq* face1 = cvHaarDetectObjects(smooth1, cascade_face, storage_face1, 1.1,  2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,  cvSize(30,30));

   
	//检测两张图片是不是只有一个人脸
    if((face1->total)>1)
	{
		cout<<"The face is not only one in the first picture"<<endl;
		return;
	}
	if((face1->total)<1)
	{
		cout<<"The face is no one in the first picture"<<endl;
		return;
	}
 
        CvRect* r=(CvRect*)cvGetSeqElem(face1,0); 
        cvRectangle(img1, cvPoint(r->x,r->y), cvPoint((r->x+r->width),(r->y+r->height)), colors[3%8],3, 8, 0);

		cvSetImageROI(smooth1,cvRect(r->x,r->y, r->width,r->height));
		IplImage* face11 = cvCreateImage(cvSize(r->width,r->height), smooth1->depth,smooth1->nChannels);  
	
        cvCopy(smooth1,face11,0);
		
		IplImage* sizeface1 = cvCreateImage(cvSize(100,130), smooth1->depth,smooth1->nChannels);
		cvResize(face11, sizeface1, CV_INTER_LINEAR);

        IplImage* LBPface1 = cvCreateImage(cvSize(100,130), smooth1->depth,smooth1->nChannels);

		LBP(sizeface1,LBPface1);
        //cvShowImage( "result1", img1);//img1

	   // cvNamedWindow( "face1", CV_WINDOW_AUTOSIZE );
		//cvShowImage( "face1",LBPface1);

		

		//Image Preparation (the second image)
		IplImage* img2 =cvCreateImage(cvSize(cvRound(image2->width * scale1),cvRound(image2->height * scale1)),image2->depth,image2->nChannels);
	   cvResize(image2, img2, CV_INTER_LINEAR);
       IplImage* gray2 = cvCreateImage(cvSize(img2->width,img2->height),8,1); 
	   cvCvtColor(img2,gray2, CV_BGR2GRAY);  //Gray

	   IplImage* smooth2 = cvCreateImage(cvSize(gray2->width,gray2->height),8,1); 
	   cvSmooth(gray2,smooth2,CV_MEDIAN);//smooth
	
       cvEqualizeHist(smooth2,smooth2); //Histogram equalization

    //Detect objects if any 
   
	   CvMemStorage* storage_face2 = 0;
       storage_face2 = cvCreateMemStorage(0) ;
       cvClearMemStorage(storage_face2); 

    double t2 = (double)cvGetTickCount(); 

    CvSeq* face2 = cvHaarDetectObjects(gray2, cascade_face, storage_face2, 1.1,  2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,  cvSize(30,30));

	//检测两张图片是不是只有一个人脸
    if((face2->total)>1)
	{
		cout<<"The face is not only one in the second picture"<<endl;
		return;
	}
	if((face2->total)<1)
	{
		cout<<"The face is no one in the second picture"<<endl;
		return;
	}
 
       r=(CvRect*)cvGetSeqElem(face2,0); 
        cvRectangle(img2, cvPoint(r->x,r->y), cvPoint((r->x+r->width),(r->y+r->height)), colors[3%8],3, 8, 0);

		cvSetImageROI(smooth2,cvRect(r->x,r->y, r->width,r->height));
		IplImage* face22 = cvCreateImage(cvSize(r->width,r->height), smooth1->depth,smooth1->nChannels);  
	    
        cvCopy(smooth2,face22,0);

		IplImage* sizeface2 = cvCreateImage(cvSize(100,130), smooth2->depth,smooth2->nChannels);
		cvResize(face22, sizeface2, CV_INTER_LINEAR);

        IplImage* LBPface2 = cvCreateImage(cvSize(100,130), smooth2->depth,smooth2->nChannels);

		LBP(sizeface2,LBPface2);
       // cvShowImage( "result2", img2);//img1

	   // cvNamedWindow( "face2", CV_WINDOW_AUTOSIZE );
		//cvShowImage( "face2",LBPface2);

		CutImage(LBPface1,LBPface2);

		int HistogramBins = 256;  
        float HistogramRange1[2]={0,255};  
        float *HistogramRange[1]={&HistogramRange1[0]};  
    

    CvHistogram *Histogram1 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
    CvHistogram *Histogram2 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	
	 cvReleaseImage(&img1);  cvReleaseImage(&img2); 
     cvReleaseImage(&gray1);  cvReleaseImage(&gray2); 
     cvReleaseImage(&face11); cvReleaseImage(&face22);  
}

//获取跳变
int getHopCount(uchar i)  
{  
    int a[8]={0};  
    int k=7;  
    int cnt=0;  
    while(i)  
    {  
        a[k]=i&1;  
        i>>=1;  
        --k;  
    }  
    for(int k=0;k<8;++k)  
    {  
        if(a[k]!=a[k+1==8?0:k+1])  
        {  
            ++cnt;  
        }  
    }  
    return cnt;  
}  
  
//降维
void lbp59table(uchar* table)  
{  
    memset(table,0,256);  
    uchar temp=1;  
    for(int i=0;i<256;++i)  
    {  
        if(getHopCount(i)<=2)  
        {  
            table[i]=temp;  
            temp++;  
        }  
        // printf("%d\n",table[i]);  
    }  
}

 //LBP算法
void LBP(IplImage* src, IplImage* dst)  
{  
    int width=src->width;  
    int height=src->height;  
    uchar table[256];  
    lbp59table(table);  
    for(int j=1;j<width-1;j++)  
    {  
        for(int i=1;i<height-1;i++)  
        {  
            uchar neighborhood[8]={0};  
            neighborhood[7] = CV_IMAGE_ELEM( src, uchar, i-1, j-1);  
            neighborhood[6] = CV_IMAGE_ELEM( src, uchar, i-1, j);  
            neighborhood[5] = CV_IMAGE_ELEM( src, uchar, i-1, j+1);  
            neighborhood[4] = CV_IMAGE_ELEM( src, uchar, i, j+1);  
            neighborhood[3] = CV_IMAGE_ELEM( src, uchar, i+1, j+1);  
            neighborhood[2] = CV_IMAGE_ELEM( src, uchar, i+1, j);  
            neighborhood[1] = CV_IMAGE_ELEM( src, uchar, i+1, j-1);  
            neighborhood[0] = CV_IMAGE_ELEM( src, uchar, i, j-1);  
            uchar center = CV_IMAGE_ELEM( src, uchar, i, j);  
            uchar temp=0;  
  
            for(int k=0;k<8;k++)  
            {  
                temp+=(neighborhood[k]>=center)<<k;  
            }  
            //CV_IMAGE_ELEM( dst, uchar, i, j)=temp;  
            CV_IMAGE_ELEM( dst, uchar, i, j)=table[temp];  
        }  
    }  
}  


void CutImage(IplImage*image1,IplImage*image2)
{
	
	int no=5;//等分
    
	//五分
	if(no==5)
	{
		IplImage* a[25];//改
		IplImage* b[25];//改
		int ceil_height =(image1->height)/no;  
		int ceil_width  =(image1->width)/no;
; 
		int n=0;

		for(int i=0;i<no*no;i++)
		{
			a[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image1->depth,image1->nChannels); 
			b[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image2->depth,image2->nChannels);
		}
		for(int i=0;i<no;i++)
		{
			for(int j=0;j<no;j++)
			{
				CvRect rect=cvRect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				cvSetImageROI(image1,rect);
				cvCopy(image1,a[n],0);
				cvSetImageROI(image2,rect);
				cvCopy(image2,b[n],0);
			   n++;
			}
		}

		cvResetImageROI(image1);
		cvResetImageROI(image2);

		int HistogramBins = 256;  
		 float HistogramRange1[2]={0,255};  
		 float *HistogramRange[1]={&HistogramRange1[0]};
		double nn;
	

	 CvHistogram *Histogram1[25];//改
	 CvHistogram *Histogram2[25];//改

	 for(int i=0;i<no*no;i++)
	{
		Histogram1[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
		Histogram2[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	}

	   double per=0;
		for(int i=0;i<no*no;i++)
		{
		
  
		  //calculate the Histogram
		 cvCalcHist(&a[i], Histogram1[i]);  
		 cvCalcHist(&b[i], Histogram2[i]);  
  
		 //normalization
		 cvNormalizeHist(Histogram1[i], 1);  
		  cvNormalizeHist(Histogram2[i], 1);

			//计算差异
		   	
			if(i<5||i%5==0||i%5==4)
			{
			  nn=0.3*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			  per+=nn;
			}
			else
			if(i==16||i==18)
			  {
			   nn=1.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
			  }
		     else
			  {
				    nn=3.4*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
				  per+=nn;
			   }

		}
		cout<<100-per<<endl;
	
		if((100-per)>persent)
		{
			persent=100-per;
			end1=img;
		}
		
		for(int i=0;i<no*no;i++)
		{
			cvReleaseImage( &a[i] );
			cvReleaseImage( &b[i] );
		}
	}

	//四分
	if(no==4)
	{
		IplImage* a[16];//改
		IplImage* b[16];//改
		int ceil_height =(image1->height)/no;  
		int ceil_width  =(image1->width)/no;
; 
		int n=0;

		for(int i=0;i<no*no;i++)
		{
			a[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image1->depth,image1->nChannels); 
			b[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image2->depth,image2->nChannels);
		}
		for(int i=0;i<no;i++)
		{
			for(int j=0;j<no;j++)
			{
				CvRect rect=cvRect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				cvSetImageROI(image1,rect);
				cvCopy(image1,a[n],0);
				cvSetImageROI(image2,rect);
				cvCopy(image2,b[n],0);
			   n++;
			}
		}

		cvResetImageROI(image1);
		cvResetImageROI(image2);

		int HistogramBins = 256;  
		 float HistogramRange1[2]={0,255};  
		 float *HistogramRange[1]={&HistogramRange1[0]};
		double nn;
	

	 CvHistogram *Histogram1[16];//改
	 CvHistogram *Histogram2[16];//改

	 for(int i=0;i<no*no;i++)
	{
		Histogram1[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
		Histogram2[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	}

	   double per=0;
		for(int i=0;i<no*no;i++)
		{
		
  
		  //calculate the Histogram
		 cvCalcHist(&a[i], Histogram1[i]);  
		 cvCalcHist(&b[i], Histogram2[i]);  
  
		 //normalization
		 cvNormalizeHist(Histogram1[i], 1);  
		  cvNormalizeHist(Histogram2[i], 1);

			//计算差异
		   	
			/*if(i<5||i%5==0||i%5==4)
			{
			  nn=0.3*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			  per+=nn;
			}
			else
			if(i==16||i==18)
			  {
			   nn=1.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
			  }
		     else
			  {
				    nn=3.4*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
				  per+=nn;
			   }*/
		  if(i<4)
		  {
			   nn=0.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }
		  else
		  if(i%4==0||i%4==3)
		  {
			   nn=2.0*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }
		  else
		  {
			   nn=3.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }

		}
		cout<<100-per<<endl;
	
		if((100-per)>persent)
		{
			persent=100-per;
			end1=img;
		}
		
		for(int i=0;i<no*no;i++)
		{
			cvReleaseImage( &a[i] );
			cvReleaseImage( &b[i] );
		}
	}

	//三分
	if(no==3)
	{
		IplImage* a[9];//改
		IplImage* b[9];//改
		int ceil_height =(image1->height)/no;  
		int ceil_width  =(image1->width)/no;
; 
		int n=0;

		for(int i=0;i<no*no;i++)
		{
			a[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image1->depth,image1->nChannels); 
			b[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image2->depth,image2->nChannels);
		}
		for(int i=0;i<no;i++)
		{
			for(int j=0;j<no;j++)
			{
				CvRect rect=cvRect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				cvSetImageROI(image1,rect);
				cvCopy(image1,a[n],0);
				cvSetImageROI(image2,rect);
				cvCopy(image2,b[n],0);
			   n++;
			}
		}

		cvResetImageROI(image1);
		cvResetImageROI(image2);

		int HistogramBins = 256;  
		 float HistogramRange1[2]={0,255};  
		 float *HistogramRange[1]={&HistogramRange1[0]};
		double nn;
	

	 CvHistogram *Histogram1[9];//改
	 CvHistogram *Histogram2[9];//改

	 for(int i=0;i<no*no;i++)
	{
		Histogram1[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
		Histogram2[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	}

	   double per=0;
		for(int i=0;i<no*no;i++)
		{
		
  
		  //calculate the Histogram
		 cvCalcHist(&a[i], Histogram1[i]);  
		 cvCalcHist(&b[i], Histogram2[i]);  
  
		 //normalization
		 cvNormalizeHist(Histogram1[i], 1);  
		  cvNormalizeHist(Histogram2[i], 1);

			//计算差异
		   	
			/*if(i<5||i%5==0||i%5==4)
			{
			  nn=0.3*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			  per+=nn;
			}
			else
			if(i==16||i==18)
			  {
			   nn=1.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
			  }
		     else
			  {
				    nn=3.4*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
				  per+=nn;
			   }*/
		  if(i==6||i==8)
		  {
			   nn=1.0*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }
		  else
		  if(i==4||i==7)
		  {
			   nn=4.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }
		  else
		  {
			   nn=2.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
			   per+=nn;
		  }

		}
		cout<<100-per<<endl;
	
		if((100-per)>persent)
		{
			persent=100-per;
			end1=img;
		}
		
		for(int i=0;i<no*no;i++)
		{
			cvReleaseImage( &a[i] );
			cvReleaseImage( &b[i] );
		}
	}

	//二分
	if(no==2)
	{
		IplImage* a[4];//改
		IplImage* b[4];//改
		int ceil_height =(image1->height)/no;  
		int ceil_width  =(image1->width)/no;
; 
		int n=0;

		for(int i=0;i<no*no;i++)
		{
			a[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image1->depth,image1->nChannels); 
			b[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image2->depth,image2->nChannels);
		}
		for(int i=0;i<no;i++)
		{
			for(int j=0;j<no;j++)
			{
				CvRect rect=cvRect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				cvSetImageROI(image1,rect);
				cvCopy(image1,a[n],0);
				cvSetImageROI(image2,rect);
				cvCopy(image2,b[n],0);
			   n++;
			}
		}

		cvResetImageROI(image1);
		cvResetImageROI(image2);

		int HistogramBins = 256;  
		 float HistogramRange1[2]={0,255};  
		 float *HistogramRange[1]={&HistogramRange1[0]};
		double nn;
	

	 CvHistogram *Histogram1[4];//改
	 CvHistogram *Histogram2[4];//改

	 for(int i=0;i<no*no;i++)
	{
		Histogram1[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
		Histogram2[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	}

	   double per=0;
		for(int i=0;i<no*no;i++)
		{
		
  
		  //calculate the Histogram
		 cvCalcHist(&a[i], Histogram1[i]);  
		 cvCalcHist(&b[i], Histogram2[i]);  
  
		 //normalization
		 cvNormalizeHist(Histogram1[i], 1);  
		  cvNormalizeHist(Histogram2[i], 1);

		//计算差异
		  nn=5.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
		  per+=nn;
		  

		}
		cout<<100-per<<endl;
	
		if((100-per)>persent)
		{
			persent=100-per;
			end1=img;
		}
		
		for(int i=0;i<no*no;i++)
		{
			cvReleaseImage( &a[i] );
			cvReleaseImage( &b[i] );
		}
	}

		//一分
	if(no==1)
	{
		IplImage* a[1];//改
		IplImage* b[1];//改
		int ceil_height =(image1->height)/no;  
		int ceil_width  =(image1->width)/no;
; 
		int n=0;

		for(int i=0;i<no*no;i++)
		{
			a[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image1->depth,image1->nChannels); 
			b[i] = cvCreateImage(cvSize(ceil_width,ceil_height), image2->depth,image2->nChannels);
		}
		for(int i=0;i<no;i++)
		{
			for(int j=0;j<no;j++)
			{
				CvRect rect=cvRect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				cvSetImageROI(image1,rect);
				cvCopy(image1,a[n],0);
				cvSetImageROI(image2,rect);
				cvCopy(image2,b[n],0);
			   n++;
			}
		}

		cvResetImageROI(image1);
		cvResetImageROI(image2);

		int HistogramBins = 256;  
		 float HistogramRange1[2]={0,255};  
		 float *HistogramRange[1]={&HistogramRange1[0]};
		double nn;
	

	 CvHistogram *Histogram1[1];//改
	 CvHistogram *Histogram2[1];//改

	 for(int i=0;i<no*no;i++)
	{
		Histogram1[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
		Histogram2[i]=cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  

	}

	   double per=0;
		for(int i=0;i<no*no;i++)
		{
		
  
		  //calculate the Histogram
		 cvCalcHist(&a[i], Histogram1[i]);  
		 cvCalcHist(&b[i], Histogram2[i]);  
  
		 //normalization
		 cvNormalizeHist(Histogram1[i], 1);  
		  cvNormalizeHist(Histogram2[i], 1);

		//计算差异
		  nn=7.5*cvCompareHist(Histogram1[i], Histogram2[i], CV_COMP_CHISQR);
		  per+=nn;
		  

		}
		cout<<100-per<<endl;
	
		if((100-per)>persent)
		{
			persent=100-per;
			end1=img;
		}
		
		for(int i=0;i<no*no;i++)
		{
			cvReleaseImage( &a[i] );
			cvReleaseImage( &b[i] );
		}
	}

}
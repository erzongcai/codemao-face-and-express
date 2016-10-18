#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv/ml.h>
#include <cstring> 
#include <cmath> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>
#include<fstream>
using namespace std;
using namespace cv;

void train();//ѵ������
void Test();//ʶ����
void face();//��⺯��

IplImage* face2;//ȫ�ֱ��� ������������

//�������������
const char* cascade_face_name = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
CvHaarClassifierCascade* cascade_face = (CvHaarClassifierCascade*)cvLoad( cascade_face_name, 0, 0, 0 );

ofstream predict_txt( "D:/SVMOUT.txt" );//��Ԥ�����洢������ı���  
string path;

void train()//ѵ������ʶ����
{

    vector<string> img_path;//�����������·��   
    vector<int> img_catg;   //����������
    int nLine = 0;    
    string buf;    
    ifstream svm_data( "D:/example.txt" );//ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�     
    unsigned long n;     
    while( svm_data )//��ѵ�������ļ����ζ�ȡ����    
    {    
        if( getline( svm_data, buf ) )    
        {    
            nLine ++;    
            if( nLine % 2 == 0 )//ע����������ͼƬȫ·����ż�����Ǳ�ǩ 
            {    
                 img_catg.push_back( atoi( buf.c_str() ) );//atoi���ַ���ת�������ͣ���־(0,1��2��...��9)��ע����������Ҫ��������𣬷�������    
            }    
            else    
            {    
                img_path.push_back( buf );//ͼ��·��    
            }    
        }    
    }    
    svm_data.close();//�ر��ļ� 


    CvMat *data_mat;//����ѵ��������HOG���� ÿ������һ�� ÿһ������һ��
	CvMat *res_mat; //����ѵ���������ÿ������һ��   
    int nImgNum = nLine / 2; //nImgNum������������ֻ���ı�������һ�룬��һ���Ǳ�ǩ  

    data_mat = cvCreateMat( nImgNum, 324, CV_32FC1 );  //��һ������Ϊ������ �ڶ���Ϊ������
    cvSetZero( data_mat );    
     
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );    //���;���,�洢ÿ�����������ͱ�־  
    cvSetZero( res_mat );  

    IplImage* src;    
    IplImage* trainImg=cvCreateImage(cvSize(28,28),8,3);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��28*28��С���������涨����324�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������    
    
    //����HOG����  
    for( string::size_type i = 0; i != img_path.size(); i++ )    
    {    
            src=cvLoadImage(img_path[i].c_str(),1);    
            if( src == NULL )    
            {    
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;    
                continue;    
            }    
    
            cout<<" ���� "<<img_path[i].c_str()<<endl;    
                   
            cvResize(src,trainImg);   //ͳһͼƬ�ߴ�

            HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);//����HOG��
            vector<float>descriptors;//��Ž��     
            hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //Hog��������      
            cout<<"HOG dims: "<<descriptors.size()<<endl;       //�������ά�� 

            n=0; 
            for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
            {    
                cvmSet(data_mat,i,n,*iter);//�洢HOG���� 
                n++;    
            }       
            cvmSet( res_mat, i, 0, img_catg[i] );    
            cout<<" �������: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;    
    }    
                
    CvSVM svm; //�½�һ��SVM      
    CvSVMParams param;//������SVMѵ����ز���  
    CvTermCriteria criteria;      
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );          
	
	svm.train( data_mat, res_mat, NULL, NULL, param );//ѵ������     

	//����ѵ���õķ�����      
	svm.save( "HOG_SVM_DATA.xml" );  
}
void face()
{
	//�������    
	/*cout<<"������һ�ž�̬ͼƬ�ĵ�ַ �� ";
    char s[100];  cin>>s;
	char* filename = s;*/
	string buf; 
	vector<string> img_tst_path;
	ifstream img_tst( "D:/SVMIN.txt" );  //������ҪԤ���ͼƬ���ϣ�����ı����ŵ���ͼƬȫ·������Ҫ��ǩ
	while( img_tst )  
	{  
		if( getline( img_tst, buf ) )  
		{  
			img_tst_path.push_back( buf );  
		}  
	}  
	img_tst.close(); 
    

	for(int x=0;x<img_tst_path.size();x++)
	{
		 IplImage *test;
		 path=img_tst_path[x];
		 test = cvLoadImage(	img_tst_path[x].c_str(), 1); //���ش�Ԥ��ͼƬ
		   if (!test)
		   {
			  cout<<"can not open"<<endl;
			  return ;
			 }
			//ͼƬԤ����
			IplImage* mini ;
		if(test->height<=400)
		{
		    mini =cvCreateImage(cvSize(cvRound(test->width),cvRound(test->height)),test->depth,test->nChannels);
		  cvResize(test, mini, CV_INTER_LINEAR);
		}
		else
		{
		  double scale = 400.0/(test ->width);
		  mini =cvCreateImage(cvSize(cvRound(test->width * scale),cvRound(test->height * scale)),test->depth,test->nChannels);
		   cvResize(test, mini, CV_INTER_LINEAR);
		}

	    IplImage* gray = cvCreateImage(cvSize(mini->width,mini->height),8,1); 
		cvCvtColor(mini,gray, CV_BGR2GRAY);  //Gray
	
		IplImage* smooth = cvCreateImage(cvSize(mini->width,mini->height),8,1); 
		cvSmooth(gray,smooth,CV_MEDIAN);//smooth

		cvEqualizeHist(smooth,smooth); //Histogram equalization

		 //Detect objects if any 
   
		 CvMemStorage* storage_face = 0;
		  storage_face = cvCreateMemStorage(0) ;
		 cvClearMemStorage(storage_face); 


	    CvSeq* face1 = cvHaarDetectObjects(smooth, cascade_face, storage_face, 1.1,  3, CV_HAAR_DO_CANNY_PRUNING,  cvSize(20,20));

   
		//���ͼƬ�ǲ���ֻ��һ������
		 if((face1->total)>1)
		{
			cout<<"The face is not only one in the  picture"<<endl;
			return;
		}
		if((face1->total)<1)
		{
			cout<<"The face is not exit"<<endl;
			return;
		}

		CvRect* r=(CvRect*)cvGetSeqElem(face1,0); 
   
		//��ͼƬ�пٳ�����
		 cvSetImageROI(smooth,cvRect(r->x,r->y, r->width,r->height));
		 face2 = cvCreateImage(cvSize(r->width,r->height), smooth->depth,smooth->nChannels);  
	
		 cvCopy(smooth,face2,0);

		cvResetImageROI(smooth);
		cvReleaseImage(&smooth);
		cvReleaseImage(&gray);
		cvReleaseImage(&test);

		Test();//����ʶ��
	}
}
void Test()//����ʶ��
{
	 CvSVM svm ;
     svm.load("D:/HOG_SVM_DATA.xml");//����ѵ���õ�xml�ļ�������ѵ��������������

     IplImage* trainTempImg=cvCreateImage(cvSize(28,28),8,1);
     cvZero(trainTempImg);    
     cvResize(face2,trainTempImg);     
     HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);      
     vector<float>descriptors;//��Ž��       
     hog->compute(trainTempImg, descriptors,Size(1,1), Size(0,0)); //Hog��������      
    // cout<<"HOG dims: "<<descriptors.size()<<endl;  //��ӡHog����ά��  ��������324 ���Բ���ʾ
     CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);   
     int n=0;    
     for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
     {    
           cvmSet(SVMtrainMat,0,n,*iter);    
           n++;    
      }   
    
      int ret = svm.predict(SVMtrainMat);//�����
	 /* if(ret==1)
	  {cout<<"the expression is happy"<<endl;}
	  if(ret==2)
	  {cout<<"the expression is sad"<<endl;}
	  if(ret==3)
	  {cout<<"the expression is surpise"<<endl;}*/

	  char result[512];
	  sprintf( result, "%s  %d\r\n",path.c_str(),ret );
       predict_txt<<result;
     
     // cvNamedWindow( "face2", CV_WINDOW_AUTOSIZE );
	  //cvShowImage( "face2",face2);
      //cvWaitKey(0);
	  cvReleaseImage(&face2);
	  cvReleaseImage(&trainTempImg);
 
}
int main()
{
	//train();
	  face();
	 // cvWaitKey(0);
	
	
}

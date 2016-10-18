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

void train();//训练函数
void Test();//识别函数
void face();//检测函数

IplImage* face2;//全局变量 处理过后的人脸

//加载人脸检测器
const char* cascade_face_name = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
CvHaarClassifierCascade* cascade_face = (CvHaarClassifierCascade*)cvLoad( cascade_face_name, 0, 0, 0 );

ofstream predict_txt( "D:/SVMOUT.txt" );//把预测结果存储在这个文本中  
string path;

void train()//训练表情识别器
{

    vector<string> img_path;//存放样本绝对路径   
    vector<int> img_catg;   //存放样本类别
    int nLine = 0;    
    string buf;    
    ifstream svm_data( "D:/example.txt" );//训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件     
    unsigned long n;     
    while( svm_data )//将训练样本文件依次读取进来    
    {    
        if( getline( svm_data, buf ) )    
        {    
            nLine ++;    
            if( nLine % 2 == 0 )//注：奇数行是图片全路径，偶数行是标签 
            {    
                 img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错    
            }    
            else    
            {    
                img_path.push_back( buf );//图像路径    
            }    
        }    
    }    
    svm_data.close();//关闭文件 


    CvMat *data_mat;//存入训练样本的HOG特征 每个样本一行 每一个特征一列
	CvMat *res_mat; //存入训练样本类别，每个样本一行   
    int nImgNum = nLine / 2; //nImgNum是样本数量，只有文本行数的一半，另一半是标签  

    data_mat = cvCreateMat( nImgNum, 324, CV_32FC1 );  //第一个参数为样本数 第二个为特征数
    cvSetZero( data_mat );    
     
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );    //类型矩阵,存储每个样本的类型标志  
    cvSetZero( res_mat );  

    IplImage* src;    
    IplImage* trainImg=cvCreateImage(cvSize(28,28),8,3);//需要分析的图片，这里默认设定图片是28*28大小，所以上面定义了324，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行    
    
    //处理HOG特征  
    for( string::size_type i = 0; i != img_path.size(); i++ )    
    {    
            src=cvLoadImage(img_path[i].c_str(),1);    
            if( src == NULL )    
            {    
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;    
                continue;    
            }    
    
            cout<<" 处理： "<<img_path[i].c_str()<<endl;    
                   
            cvResize(src,trainImg);   //统一图片尺寸

            HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);//建立HOG类
            vector<float>descriptors;//存放结果     
            hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //Hog特征计算      
            cout<<"HOG dims: "<<descriptors.size()<<endl;       //输出特征维数 

            n=0; 
            for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
            {    
                cvmSet(data_mat,i,n,*iter);//存储HOG特征 
                n++;    
            }       
            cvmSet( res_mat, i, 0, img_catg[i] );    
            cout<<" 处理完毕: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;    
    }    
                
    CvSVM svm; //新建一个SVM      
    CvSVMParams param;//这里是SVM训练相关参数  
    CvTermCriteria criteria;      
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );          
	
	svm.train( data_mat, res_mat, NULL, NULL, param );//训练数据     

	//保存训练好的分类器      
	svm.save( "HOG_SVM_DATA.xml" );  
}
void face()
{
	//检测样本    
	/*cout<<"请输入一张静态图片的地址 ： ";
    char s[100];  cin>>s;
	char* filename = s;*/
	string buf; 
	vector<string> img_tst_path;
	ifstream img_tst( "D:/SVMIN.txt" );  //加载需要预测的图片集合，这个文本里存放的是图片全路径，不要标签
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
		 test = cvLoadImage(	img_tst_path[x].c_str(), 1); //加载待预测图片
		   if (!test)
		   {
			  cout<<"can not open"<<endl;
			  return ;
			 }
			//图片预处理
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

   
		//检测图片是不是只有一个人脸
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
   
		//从图片中抠出人脸
		 cvSetImageROI(smooth,cvRect(r->x,r->y, r->width,r->height));
		 face2 = cvCreateImage(cvSize(r->width,r->height), smooth->depth,smooth->nChannels);  
	
		 cvCopy(smooth,face2,0);

		cvResetImageROI(smooth);
		cvReleaseImage(&smooth);
		cvReleaseImage(&gray);
		cvReleaseImage(&test);

		Test();//表情识别
	}
}
void Test()//表情识别
{
	 CvSVM svm ;
     svm.load("D:/HOG_SVM_DATA.xml");//加载训练好的xml文件，这里训练的是人脸表情

     IplImage* trainTempImg=cvCreateImage(cvSize(28,28),8,1);
     cvZero(trainTempImg);    
     cvResize(face2,trainTempImg);     
     HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);      
     vector<float>descriptors;//存放结果       
     hog->compute(trainTempImg, descriptors,Size(1,1), Size(0,0)); //Hog特征计算      
    // cout<<"HOG dims: "<<descriptors.size()<<endl;  //打印Hog特征维数  ，这里是324 可以不显示
     CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);   
     int n=0;    
     for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
     {    
           cvmSet(SVMtrainMat,0,n,*iter);    
           n++;    
      }   
    
      int ret = svm.predict(SVMtrainMat);//检测结果
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

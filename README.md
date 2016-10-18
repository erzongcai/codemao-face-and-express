# codemao-face-and-express
100steps project make by a group with my classmates
    在各种生物特征识别方法中，人脸有其自身特殊的优势，因而在生物识别中有着重要的地位。本小组这次百步梯攀登计划的课题就是围绕人脸这一生物特征而展开的研究，包括以下两个方面：
    人脸识别。人脸识别即面部识别，是基于人的脸部特征信息进行身份识别的一种生物识别技术。自上个世纪80年代以来，人脸识别技术的理论研究一直不断发展，各种不同的算法相继被提出。根据特征形式的不同，我们将人脸识别方法主要分为三类：基于局部特征、基于全局特征、基于混合特征。本次研究采用局部二值模式（LBP）方法，实现不同人脸相似度的计算。
    表情识别。心理学家 Mehrabian 提出，在人们的交流中，只有7%的信息是通过语言来传递，而通过面部表情传递的信息量却达到了 55%。随着人机交互与情感计算技术的快速发展 , 人脸表情识别已成为人们 研究的热点 。同人脸识别一样，表情识别在上世纪就开始有了研究，识别算法和人脸识别的算法也差不多相似，有主成分分析 、 独立分量分析 、 Fisher 线性判别分析等方法。本次研究采用方向梯度直方图（HOG）和支持向量机（SVM）相结合的方法，实现对人脸三种主要表情的判断。
    人脸识别，表情识别的实现过程基本相似，都按照：人脸检测，人脸处理，特征提取，特征识别这样的流程实现。
    对于人脸识别，其主要研究背景是在于身份的鉴定上。身份鉴定是人类社会日常生活中的基本活动之一，人们几乎每时每刻都需要证明自己的身份。关于个人身份鉴定的问题可以分为两类：认证(Verification)和辨识(Identification)。“认证”指的是验证用户是否为他所声明的身份，“辨识”指的是确定当前用户的身份。传统的个人身份鉴定的方法主要依靠信物(如各种证件、钥匙、磁卡等)或身份标识信息(如口令和密码)，信物携带不便且容易丢失、被盗、损坏；身份标识信息容易遗忘、被他人窃取或破解；更为严重的是传统身份认证方法往往无法区分信物或身份标识信息真正的拥有者和冒充者。一旦他人获得信物或身份标识信息就具有与拥有者相同的权力，使真正拥有者的利益受到威胁。显然，这些致命的缺点使得传统的身份鉴定方法已经完全不能满足现代社会的要求，于是人们亟需寻找一种更方便、更可靠、更安全的身份验证方式。生物识别技术正是在这样的需求下应运而生的。
    对于生物特征识别，相比于依靠信物来识别更加具有以下的特性： 
        ①	普遍性：每个正常人都应该具有这种特征； 
        ②惟一性：不同的人应该具有各不相同的特征； 
        ③可采集性：所选择的特征可以定量测量； 
        ④稳定性：所选择的特征至少在一段较长的时间内是不变的，并且特征的采集不随条件、环境的变化而变化。 
        ⑤安全性：用欺诈的方法骗过系统的难易程度； 
        ⑥理论依据：是否具有相关的、可信的研究背景作为技术支持； 
    作为生物特征识别的一个分支， 人脸识别在罪犯身份验证、安全验证、信用卡验证、银行和海关的实时监控等方面具有广阔的应用前景。 由于其非入侵性和用户友好性， 人脸识别一直是模式识别和计算机视觉领域的热点课题。

    对于表情识别来说，主要研究背景是在于情感方面。人们对于人脸表情的研究可以追溯到 19 世纪 , 生物学家 Darwin 在《人类和动物的表情》 一书中 , 就对人类的面部表情与动物的面部表情进行了研究和比较。心理学家 Mehrabian 提出, 在人们的交流中, 只有 7 % 的信息是通过语言来传递, 而通过面部表情传递的信息量却达到了 55 % 。人机交互模式已从语言命令交互阶段 、 图像用户界面交互阶段发展到自 然和谐的人机交互阶段 。 同时, 由麻省理工学院Picard 教授提出的情感计算 (affectivecomputing) 领域正蓬勃兴起 , 其目 标是使计算机拥有情感 , 即能够像人类一样识别和表达情感 ,使人机交互更加人性化 。 为了使人机交互更加和谐与自然 , 新型的人机交互技术正逐渐成为研究热点 。
    人脸表情识别是人机交互与情感计算研究的重要组成部分 。 由于人脸表情包含丰富的行为信息 , 因此对人脸表情进行识别有利于了解人类的情感等理状态 , 并可进行有效的人机交互 。 人脸表情识别涉及心理学 、 社会学、人类学、生命科学、 认知科学、 生物学、 病理学、计算机科学等研究领域。 可见, 人脸表情识别的进展对提高人工情感智能水平和探索人类情感及认知能力极具科学意义 , 并将促进相关学科的发展 。 

**此代码会在其他电脑上运行,不允许在本地配置环境或编译,目前的代码已经保证可以在其他电脑上编译运行**
增加config的mode选项中增加"calibration"模式，在原有的sync.cpp基础上做增量化修改。开启"calibration"模式时，需要调用奥比中光获取相机内参/畸变参数的SDK，利用opencv库中函数求解相机外参。实现细节如下
## 实现细节
需要修改sync.cpp中从init_extrinsic_path读取相机外参的逻辑：从文件中读取的相机外参旋转不再以欧拉角的形式表示，而是以旋转矩阵的形式表示。外参规定为世界系到相机系到变换。
所有的交互式输入输出阶段都需要支持输入q退出程序
1. 进入calibration模式后，在命令行进行交互。首先在命令行输出：
==========calibration mode(press "q" for quitting)==========
choose calibration method:
    1. chessboard
    2. block
然后等待用户输入，根据输入选择对应的标定方式。chessboard表示利用棋盘格进行标定，block表示利用标定块进行标定（标定块标定现阶段不要求实现，但在代码中需要预留实现的接口）。加下来介绍棋盘格标定的交互实现细节。
2. 相机两两标定的交互式命令实现。用户在开始界面输入1进入棋盘格标定后，需要用户输入当前标定的两台相机序号，序号对应config文件中相机同步配置中的序号（例如"00"、"01"）输出提示示例如下：
==========chessboard calibration(press "q" for quitting)==========
    index of first camera(e.g. "00"): (等待用户输入后,输出下一行提示)
    index of second camera(e.g. "01"): (等待用户输入)
在用户输入完要进行标定的两台相机后，程序需要调用奥比中光SDK中的API，获取相机的内参/畸变参数，并为序号对应的两台相机创建RGBD流（需保证相机同步），为后续的标定做准备。两个相机标定完成后，再次输出提示示例如下，标定剩余的多台相机，提示中需要显示已经标定完成的相机。
==========chessboard calibration(press "q" for quitting)==========
camera calibrated: 
    (first1,second1)
    (first2,second2)
    ...
press "c" to calculate final calibration result

    index of first camera(e.g. "00"): (等待用户输入后,输出下一行提示)
    index of second camera(e.g. "01"): (等待用户输入)
当用户输入的两个相机（不区分顺序）已经在之前标定过之后，需要进行提示，并让用户重新选择相机。当用户输入c后，结合所有已两两标定的相机，以"00"号相机的相机系作为世界坐标系，计算所有相机的外参（第一个相机到所有相机的变换），并将结果保存到config文件中的init_extrinsic_path所指向的文件中，保存格式参考src/sync/extrinsic.json，但是要将旋转用旋转矩阵保存
3. 相机两两标定实现细节。在用户输入要标定的两个相机后，进入交互式采样界面：
==========chessboard calibration(press "q" for quitting)==========
calibrating cam_idx1 and cam_idx2  
    press "1" to sample images
    press "0" to calculate relative extrinsic
用户输入1时，从两个相机的流中同步获取一帧图像，并输出两个相机捕获的image对应的timestep：
captured image:
    camera_idx1: timestep1,  camera_idx2: timestep2
获取的图像不需要保存，暂存在内存中，用于后续opencv计算外参。用户输入多次1，获取多对图像后，输入0，调用opencv库中工具，计算第一个相机到第二个相机的变换，即以第一个相机坐标系为世界坐标系，第二个相机的外参。计算完成后，返回2中相机选择界面。重复上述过程，直到在相机选择界面中输入"c"，计算最终结果。
4. 在符合本文档基本设计要求的前提下保证界面简洁美观，容易混淆的地方使用不同颜色字体区分（例如2中列举已标定相机）。优化log排版逻辑，例如3中，每次按下1，log中的captured image信息应该打印在press "1" to sample images提示的上方。如果opencv在计算外参时，需要输入本文档未提及的配置信息，需要在config文件中加入配置的接口，并在对话框中介绍接口中应该填写什么配置。对sync.cpp的修改不得影响其他mode的功能。

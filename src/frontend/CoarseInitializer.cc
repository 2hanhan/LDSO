#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"
#include "internal/GlobalFuncs.h"
#include "frontend/PixelSelector2.h"
#include "frontend/CoarseInitializer.h"
#include "frontend/nanoflann.h"

#include <iostream>

namespace ldso
{

    CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3())
    {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
        {
            points[lvl] = 0;
            numPoints[lvl] = 0;
        }

        JbBuffer = new Vec10f[ww * hh];
        JbBuffer_new = new Vec10f[ww * hh];

        fixAffine = true;
        printDebug = false;

        wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
        wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
        wM.diagonal()[6] = SCALE_A;
        wM.diagonal()[7] = SCALE_B;
    }

    CoarseInitializer::~CoarseInitializer()
    {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
        {
            if (points[lvl] != 0)
                delete[] points[lvl];
        }

        delete[] JbBuffer;
        delete[] JbBuffer_new;
    }

    /**
     * @brief 对第0帧进行跟踪
     *
     * @param newFrameHessian
     * @return true 位移足够大，然后再优化5帧
     * @return false
     */
    bool CoarseInitializer::trackFrame(shared_ptr<FrameHessian> newFrameHessian)
    {

        newFrame = newFrameHessian;
        int maxIterations[] = {5, 5, 10, 30, 50};

        alphaK = 2.5 * 2.5;
        alphaW = 150 * 150;
        regWeight = 0.8;
        couplingWeight = 1;

        // - 初始化每个点的逆深度为1，初始化光度参数，位姿SE3
        if (!snapped) //位移是否足骨
        {
            //初始化
            thisToNext.translation().setZero();
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
            {
                int npts = numPoints[lvl];
                Pnt *ptsl = points[lvl];
                for (int i = 0; i < npts; i++)
                {
                    ptsl[i].iR = 1;
                    ptsl[i].idepth_new = 1;
                    ptsl[i].lastHessian = 0;
                }
            }
        }

        SE3 refToNew_current = thisToNext;
        AffLight refToNew_aff_current = thisToNext_aff;

        //如果有仿射系数则估计一个初值
        if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
            refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure),
                                            0); // coarse approximation.

        Vec3f latestRes = Vec3f::Zero();
        //从顶层到底层
        for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
        {
            // - 使用计算过的上一层来初始化下一层
            //顶层未初始化到，reset来完成
            if (lvl < pyrLevelsUsed - 1)
                propagateDown(lvl + 1); //不是最顶层，使用上一层初始化下一层

            Mat88f H, Hsc;
            Vec8f b, bsc;
            resetPoints(lvl); //这里对顶层进行初始化
            // - 迭代之前计算能量，hessian等
            Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
            applyStep(lvl); //新的能量赋给旧的

            float lambda = 0.1;
            float eps = 1e-4;
            int fails = 0;

            // - 迭代求解
            int iteration = 0;
            while (true)
            {
                // 计算边缘化后的hessian
                Mat88f Hl = H;
                for (int i = 0; i < 8; i++)
                    Hl(i, i) *= (1 + lambda);
                //舒尔补，边缘化掉逆深度状态
                Hl -= Hsc * (1 / (1 + lambda)); //以为dd必定是对角线上的，所有也乘以倒数
                Vec8f bl = b - bsc * (1 / (1 + lambda));

                Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
                bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

                // 求解增量
                Vec8f inc;
                if (fixAffine) //固定光度参数
                {
                    inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                      (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                    inc.tail<2>().setZero();
                }
                else
                    inc = -(wM * (Hl.ldlt().solve(bl))); //=-H^-1 * b.

                // 更新状态，doStep中更新逆深度
                SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
                AffLight refToNew_aff_new = refToNew_aff_current;
                refToNew_aff_new.a += inc[6];
                refToNew_aff_new.b += inc[7];
                doStep(lvl, lambda, inc);

                //计算更新后的能量并与之前的对比，判断是否可行
                Mat88f H_new, Hsc_new;
                Vec8f b_new, bsc_new;
                Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
                Vec3f regEnergy = calcEC(lvl);

                float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
                float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

                bool accept = eTotalOld > eTotalNew; //能量函数是否减小

                //能量函数减小，则接受此次更新
                if (accept)
                {

                    if (resNew[1] == alphaK * numPoints[lvl]) //当 alphaEnergy  > alphak*npts
                        snapped = true;
                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    refToNew_aff_current = refToNew_aff_new;
                    refToNew_current = refToNew_new;
                    applyStep(lvl);
                    optReg(lvl);   //更新iR
                    lambda *= 0.5; //减小lambda
                    fails = 0;
                    if (lambda < 0.0001)
                        lambda = 0.0001;
                }
                else
                {
                    fails++;
                    lambda *= 4; //增大lambda
                    if (lambda > 10000)
                        lambda = 10000;
                }

                bool quitOpt = false;
                //迭代停止条件，收敛/迭代大于次数/失败2次以上
                if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
                {
                    Mat88f H, Hsc;
                    Vec8f b, bsc;

                    quitOpt = true;
                }

                if (quitOpt)
                    break;
                iteration++;
            }
            latestRes = resOld;
        }

        // 优化后赋值位姿，从底层计算上层点的深度
        thisToNext = refToNew_current;
        thisToNext_aff = refToNew_aff_current;

        for (int i = 0; i < pyrLevelsUsed - 1; i++)
            propagateUp(i);

        frameID++;
        if (!snapped)
            snappedAt = 0;

        if (snapped && snappedAt == 0)
            snappedAt = frameID; //位移足够的帧数

        // 位移足够大，然后再优化5帧
        return snapped && frameID > snappedAt + 5;
    }

    /**
     * @brief 计算能量函数、hessian矩阵，舒尔补，Schur
     * calculates residual, Hessian and Hessian-block neede for re-substituting depth.
     * @param lvl
     * @param H_out
     * @param b_out
     * @param H_out_sc
     * @param b_out_sc
     * @param refToNew
     * @param refToNew_aff
     * @param plot
     * @return Vec3f
     */
    Vec3f CoarseInitializer::calcResAndGS(
        int lvl, Mat88f &H_out, Vec8f &b_out,
        Mat88f &H_out_sc, Vec8f &b_out_sc,
        const SE3 &refToNew, AffLight refToNew_aff,
        bool plot)
    {
        int wl = w[lvl], hl = h[lvl];
        //当前层图像以及梯度
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
        Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

        // 旋转矩阵R*内存矩阵K_inv
        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();                                   //平移
        Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); //光度参数

        // 该层相机参数
        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];

        Accumulator11 E;   // 1*1的累加器
        acc9.initialize(); //初始值，分配空间
        E.initialize();

        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        for (int i = 0; i < npts; i++)
        {

            Pnt *point = ptsl + i;

            point->maxstep = 1e10;
            if (!point->isGood) //点不好
            {
                E.updateSingle((float)(point->energy[0])); //累加
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

            VecNRf dp0; // 8*1矩阵，每个点附近的残差个数为8个
            VecNRf dp1;
            VecNRf dp2;
            VecNRf dp3;
            VecNRf dp4;
            VecNRf dp5;
            VecNRf dp6;
            VecNRf dp7;
            VecNRf dd;                 //逆深度的导数
            VecNRf r;                  //残差
            JbBuffer_new[i].setZero(); // 10*1 向量

            // sum over all residuals.
            bool isGood = true;
            float energy = 0;
            for (int idx = 0; idx < patternNum; idx++)
            {
                // pattern的坐标偏移
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
                //归一化坐标 Pj
                float u = pt[0] / pt[2];
                float v = pt[1] / pt[2];
                //像素坐标Pj
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                // dpi/pz'
                float new_idepth = point->idepth_new / pt[2]; //上一帧的逆深度

                //落在边缘附近、深度小于0，则不好
                if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
                {
                    isGood = false;
                    break;
                }

                //差值得到新图像中的 patcch像素值，（输入3维，输出3维像素值 + x方向梯度 + y方向梯度）
                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                // Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

                //参考上一帧的 patch上的像素值，输出一维像素值
                // float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

                //像素值又穷，则好
                if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
                {
                    isGood = false;
                    break;
                }

                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];                   //残差
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); // huber权重
                // huberweight*(2-huberweight)*Objective Function
                // robust 权重核函数之间的关系
                energy += hw * residual * residual * (2 - hw);

                // Pj 对逆深度求导数
                float dxdd = (t[0] - t[2] * u) / pt[2];
                float dydd = (t[1] - t[2] * v) / pt[2];

                if (hw < 1)
                    hw = sqrtf(hw);
                // dxfx，dyfy
                float dxInterp = hw * hitColor[1] * fxl;
                float dyInterp = hw * hitColor[2] * fyl;
                // 残差对Tj求导数
                dp0[idx] = new_idepth * dxInterp;
                dp1[idx] = new_idepth * dyInterp;
                dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
                dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
                dp5[idx] = -v * dxInterp + u * dyInterp;
                //残差对光度参数求导数
                dp6[idx] = -hw * r2new_aff[0] * rlR;
                dp7[idx] = -hw * 1;
                //残差对 i逆深度求导数
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;
                r[idx] = hw * residual;

                //像素误差对逆深度的导数，取模倒数
                float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
                if (maxstep < point->maxstep)
                    point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.
                // 计算hessian的第一行，已经Jr关于逆深度的那一行
                //用了计算舒尔补
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];
                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

            //如果点的pattern(其中一个像素)超出图像，像素值无穷大，或者残差大于阈值
            if (!isGood || energy > point->outlierTH * 20)
            {
                E.updateSingle((float)(point->energy[0])); //上一帧的加进来
                point->isGood_new = false;
                point->energy_new = point->energy; //上一次的给当前次的
                continue;
            }

            //内点则加进能量函数
            // add into energy.
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy;

            // update Hessian matrix.
            //使用128位相当于每次加4个float数，因此是i+4
            for (int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                    _mm_load_ps(((float *)(&dp0)) + i),
                    _mm_load_ps(((float *)(&dp1)) + i),
                    _mm_load_ps(((float *)(&dp2)) + i),
                    _mm_load_ps(((float *)(&dp3)) + i),
                    _mm_load_ps(((float *)(&dp4)) + i),
                    _mm_load_ps(((float *)(&dp5)) + i),
                    _mm_load_ps(((float *)(&dp6)) + i),
                    _mm_load_ps(((float *)(&dp7)) + i),
                    _mm_load_ps(((float *)(&r)) + i));

            //一些程序应对的每次处理4个后余下的不足4个的情况
            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                acc9.updateSingle(
                    (float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
                    (float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
                    (float)r[i]);
        }

        E.finish();
        acc9.finish();

        // calculate alpha energy, and decide if we cap it.
        Accumulator11 EAlpha;
        EAlpha.initialize();
        for (int i = 0; i < npts; i++)
        {
            Pnt *point = ptsl + i;
            if (!point->isGood_new) //点不好，用之前的
            {
                E.updateSingle((float)(point->energy[1]));
            }
            else
            {
                //开始初始化都是成1
                point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
                E.updateSingle((float)(point->energy_new[1]));
            }
        }
        EAlpha.finish();                                                                       // 只计算位移是否足够大
        float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts); //平移越大越容易初始化成功

        // compute alpha opt.
        float alphaOpt;
        if (alphaEnergy > alphaK * npts) //平移大于一定值
        {
            alphaOpt = 0;
            alphaEnergy = alphaK * npts;
        }
        else
        {
            alphaOpt = alphaW;
        }

        acc9SC.initialize();
        for (int i = 0; i < npts; i++)
        {
            Pnt *point = ptsl + i;
            if (!point->isGood_new)
                continue;

            point->lastHessian_new = JbBuffer_new[i][9]; //对逆深度 dd*dd

            //前面energe加上了(d-1)*(d-1)，所以dd=1 ，r +=(d-1)
            // 位移足够大alphaOpt为0，不限制d逆深度均值为1
            JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1); // r*dd逆深度的雅克比*雅克比
            JbBuffer_new[i][9] += alphaOpt;                           //对逆深度导数为1 //dd*dd逆深度的雅克比*残差

            if (alphaOpt == 0)
            {
                JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
                JbBuffer_new[i][9] += couplingWeight;
            }

            JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]); //取逆是协方差，做权重
            //做权重，计算是舒尔补项！
            acc9SC.updateSingleWeighted(
                (float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2],
                (float)JbBuffer_new[i][3],
                (float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6],
                (float)JbBuffer_new[i][7],
                (float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
        }
        acc9SC.finish();

        H_out = acc9.H.topLeftCorner<8, 8>();       // / acc9.num;
        b_out = acc9.H.topRightCorner<8, 1>();      // / acc9.num;
        H_out_sc = acc9SC.H.topLeftCorner<8, 8>();  // / acc9.num;
        b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;

        //给 t 对应的hessian对角线加上一个数 ，b 也加上
        H_out(0, 0) += alphaOpt * npts;
        H_out(1, 1) += alphaOpt * npts;
        H_out(2, 2) += alphaOpt * npts;

        Vec3f tlog = refToNew.log().head<3>().cast<float>(); //李代书，平移部分（上一次的位姿值）
        b_out[0] += tlog[0] * alphaOpt * npts;
        b_out[1] += tlog[1] * alphaOpt * npts;
        b_out[2] += tlog[2] * alphaOpt * npts;

        return Vec3f(E.A, alphaEnergy, E.num); //能量值，?，使用的点数
    }

    float CoarseInitializer::rescale()
    {
        float factor = 20 * thisToNext.translation().norm();
        return factor;
    }

    Vec3f CoarseInitializer::calcEC(int lvl)
    {
        if (!snapped)
            return Vec3f(0, 0, numPoints[lvl]);
        AccumulatorX<2> E;
        E.initialize();
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++)
        {
            Pnt *point = points[lvl] + i;
            if (!point->isGood_new)
                continue;
            float rOld = (point->idepth - point->iR);
            float rNew = (point->idepth_new - point->iR);
            E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));
        }
        E.finish();

        return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
    }

    /**
     * @brief 使用最近的点更新每个点的iR
     *
     * @param lvl
     */
    void CoarseInitializer::optReg(int lvl)
    {
        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        // 位移不足设置iR为1
        if (!snapped)
        {
            for (int i = 0; i < npts; i++)
                ptsl[i].iR = 1;
            return;
        }

        for (int i = 0; i < npts; i++)
        {
            Pnt *point = ptsl + i;
            if (!point->isGood)
                continue;

            float idnn[10];
            int nnn = 0;
            //获取当前点周围最近的10个点，质量好的点的iR
            for (int j = 0; j < 10; j++)
            {
                if (point->neighbours[j] == -1)
                    continue;
                Pnt *other = ptsl + point->neighbours[j];
                if (!other->isGood)
                    continue;
                idnn[nnn] = other->iR;
                nnn++;
            }

            // 与最近点中位数进行加权获得新的iR
            if (nnn > 2)
            {
                std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); //获得中位数
                point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
            }
        }
    }

    void CoarseInitializer::propagateUp(int srcLvl)
    {
        assert(srcLvl + 1 < pyrLevelsUsed);
        // set idepth of target

        int nptss = numPoints[srcLvl];
        int nptst = numPoints[srcLvl + 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl + 1];

        // set to zero.
        for (int i = 0; i < nptst; i++)
        {
            Pnt *parent = ptst + i;
            parent->iR = 0;
            parent->iRSumNum = 0;
        }

        for (int i = 0; i < nptss; i++)
        {
            Pnt *point = ptss + i;
            if (!point->isGood)
                continue;

            Pnt *parent = ptst + point->parent;
            parent->iR += point->iR * point->lastHessian;
            parent->iRSumNum += point->lastHessian;
        }

        for (int i = 0; i < nptst; i++)
        {
            Pnt *parent = ptst + i;
            if (parent->iRSumNum > 0)
            {
                parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
                parent->isGood = true;
            }
        }

        optReg(srcLvl + 1);
    }

    /**
     * @brief 使用上一层信息初始化下一层
     *
     * @param srcLvl 当前层金字塔层数+1
     */
    void CoarseInitializer::propagateDown(int srcLvl)
    {
        assert(srcLvl > 0);
        // set idepth of target

        int nptst = numPoints[srcLvl - 1]; //当前层点数
        Pnt *ptss = points[srcLvl];        //上一层点集
        Pnt *ptst = points[srcLvl - 1];    //当前层点集

        for (int i = 0; i < nptst; i++)
        {
            Pnt *point = ptst + i;              //变量当前层的点
            Pnt *parent = ptss + point->parent; //找到当前点的parent

            if (!parent->isGood || parent->lastHessian < 0.1)
                continue;
            if (!point->isGood)
            {
                //当前点不好，则把parent点的值直接给它，并置位为Good
                point->iR = point->idepth = point->idepth_new = parent->iR;
                point->isGood = true;
                point->lastHessian = 0;
            }
            else
            {
                //通过hessian给point和parent加权获得新的iR
                // iR可以看着是深度值，使用高斯归一化积，hessian是信息矩阵
                float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) /
                              (point->lastHessian * 2 + parent->lastHessian);
                point->iR = point->idepth = point->idepth_new = newiR;
            }
        }
        optReg(srcLvl - 1); //当前层，使用最近点更新iR
    }

    void CoarseInitializer::makeGradients(Eigen::Vector3f **data)
    {
        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
        {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            Eigen::Vector3f *dINew_l = data[lvl];
            Eigen::Vector3f *dINew_lm = data[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++)
                    dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

            for (int idx = wl; idx < wl * (hl - 1); idx++)
            {
                dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
                dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
            }
        }
    }

    /**
     * @brief 设置初始化第一帧
     *
     * @param HCalib 相机内参
     * @param newFrameHessian
     */
    void CoarseInitializer::setFirst(shared_ptr<CalibHessian> HCalib, shared_ptr<FrameHessian> newFrameHessian)
    {

        // step 1 计算图像每层的内参
        makeK(HCalib);
        firstFrame = newFrameHessian;

        PixelSelector sel(w[0], h[0]);

        float *statusMap = new float[w[0] * h[0]];
        bool *statusMapB = new bool[w[0] * h[0]];

        float densities[] = {0.03, 0.05, 0.15, 0.5, 1}; //不同层提取像素密度
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
        {
            // step 2 根据不同的层数提取符合梯度阈值要求像素
            sel.currentPotential = 3; // pot大小
            int npts;
            if (lvl == 0)
            { // 第0层 提取
                npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
            }
            else
            { //第1层 提取
                npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);
            }

            //如果点非空，释放内参，创建新的points
            if (points[lvl] != 0)
                delete[] points[lvl];
            points[lvl] = new Pnt[npts];

            // set idepth map to initially 1 everywhere.
            int wl = w[lvl], hl = h[lvl]; //每一层的图像大小
            Pnt *pl = points[lvl];        //每一层的图像上的点
            int nl = 0;
            //要留出pattern的空间， 2border
            // step 3 在选出的像素中，添加点信息
            for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
                for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++)
                {
                    //如果是被选中的像素
                    if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0))
                    {
                        // assert(patternNum==9);
                        pl[nl].u = x + 0.1;
                        pl[nl].v = y + 0.1;
                        pl[nl].idepth = 1;
                        pl[nl].iR = 1;
                        pl[nl].isGood = true;
                        pl[nl].energy.setZero();
                        pl[nl].lastHessian = 0;
                        pl[nl].lastHessian_new = 0;
                        pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

                        Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl]; //该像素的梯度
                        float sumGrad2 = 0;
                        //计算pattern内像素的梯度和
                        for (int idx = 0; idx < patternNum; idx++)
                        {
                            int dx = patternP[idx][0]; // pattern 的偏移
                            int dy = patternP[idx][1];
                            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
                            sumGrad2 += absgrad;
                        }

                        //外点的阈值与pattern的大小有关，一个像素是12*12
                        pl[nl].outlierTH = patternNum * setting_outlierTH;

                        nl++;
                        assert(nl <= npts);
                    }
                }

            numPoints[lvl] = nl; //点的数目，去掉一些边界上的点
        }
        delete[] statusMap;
        delete[] statusMapB;

        // step 4 计算点的最近邻与parent
        makeNN();

        //参数初始化
        thisToNext = SE3();
        snapped = false; //位移是否足够
        frameID = snappedAt = 0;

        for (int i = 0; i < pyrLevelsUsed; i++)
            dGrads[i].setZero();
    }

    /**
     * @brief 重置点的energy，indepth_new参数
     *
     * @param lvl
     */
    void CoarseInitializer::resetPoints(int lvl)
    {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++)
        {
            //重置
            pts[i].energy.setZero();
            pts[i].idepth_new = pts[i].idepth;

            //如果是最顶层，则使用周围点平均值来重置
            if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood)
            {
                float snd = 0, sn = 0;
                for (int n = 0; n < 10; n++)
                {
                    if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
                        continue;
                    snd += pts[pts[i].neighbours[n]].iR;
                    sn += 1;
                }

                if (sn > 0)
                {
                    pts[i].isGood = true;
                    pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
                }
            }
        }
    }

    /**
     * @brief 求出状态增量后，计算被边缘化掉的逆深度，更新逆深度
     *
     * @param lvl
     * @param lambda
     * @param inc
     */
    void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
    {

        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++)
        {
            if (!pts[i].isGood)
                continue;

            float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
            float step = -b * JbBuffer[i][9] / (1 + lambda);

            float maxstep = maxPixelStep * pts[i].maxstep; //逆深度最大只能增加这些
            if (maxstep > idMaxStep)
                maxstep = idMaxStep;

            if (step > maxstep)
                step = maxstep;
            if (step < -maxstep)
                step = -maxstep;

            // 更新得到新的逆深度
            float newIdepth = pts[i].idepth + step;
            if (newIdepth < 1e-3)
                newIdepth = 1e-3;
            if (newIdepth > 50)
                newIdepth = 50;
            pts[i].idepth_new = newIdepth;
        }
    }

    /**
     * @brief 新的赋值给之前的（能量，点状态，逆深度，hessian）
     *
     * @param lvl
     */
    void CoarseInitializer::applyStep(int lvl)
    {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++)
        {
            if (!pts[i].isGood)
            {
                pts[i].idepth = pts[i].idepth_new = pts[i].iR;
                continue;
            }
            pts[i].energy = pts[i].energy_new;
            pts[i].isGood = pts[i].isGood_new;
            pts[i].idepth = pts[i].idepth_new;
            pts[i].lastHessian = pts[i].lastHessian_new;
        }
        std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
    }

    void CoarseInitializer::makeK(shared_ptr<CalibHessian> HCalib)
    {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level)
        {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level)
        {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    /**
     * @brief 生成每一层的KDTree，并用KDTree找到临近点和parent
     *
     */
    void CoarseInitializer::makeNN()
    {
        const float NNDistFactor = 0.05;

        // [第1个参数：distance] [第2个参数：datasetadaptor [第3个参数：维度]
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
            FLANNPointcloud, 2>
            KDTree;

        // build indices
        FLANNPointcloud pcs[PYR_LEVELS]; //每一层建立一个点云
        KDTree *indexes[PYR_LEVELS];     //点云建立KDTree
        // 每层建立一个KDTree索引的二维点云
        for (int i = 0; i < pyrLevelsUsed; i++)
        {
            pcs[i] = FLANNPointcloud(numPoints[i], points[i]); // 二维点云
            //参数：[维度] [点数据] [叶节点中最大的点数（越多build越快，query越慢）]
            indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
            indexes[i]->buildIndex();
        }

        const int nn = 10;

        // find NN & parents
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
        {
            Pnt *pts = points[lvl];
            int npts = numPoints[lvl];

            int ret_index[nn];  //搜索到的临近点
            float ret_dist[nn]; //搜索到点的距离
            //搜索结果，距离最近的nn个和1个
            nanoflann::KNNResultSet<float, int, int> resultSet(nn);
            nanoflann::KNNResultSet<float, int, int> resultSet1(1);

            for (int i = 0; i < npts; i++)
            {
                // resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
                resultSet.init(ret_index, ret_dist);
                Vec2f pt = Vec2f(pts[i].u, pts[i].v); //当前点
                //使用建立的KDTree来查询最近领
                indexes[lvl]->findNeighbors(resultSet, (float *)&pt, nanoflann::SearchParams());
                int myidx = 0;
                float sumDF = 0;
                // - 给每个点的临近点赋值
                for (int k = 0; k < nn; k++)
                {
                    pts[i].neighbours[myidx] = ret_index[k];      //最近的索引
                    float df = expf(-ret_dist[k] * NNDistFactor); //距离属于指数形式
                    sumDF += df;                                  //距离和
                    pts[i].neighboursDist[myidx] = df;
                    assert(ret_index[k] >= 0 && ret_index[k] < npts);
                    myidx++;
                }
                //对距离进行归10化
                for (int k = 0; k < nn; k++)
                    pts[i].neighboursDist[k] *= 10 / sumDF;

                // -  高一层的图像中找到的该点的parent
                if (lvl < pyrLevelsUsed - 1)
                {
                    resultSet1.init(ret_index, ret_dist);
                    pt = pt * 0.5f - Vec2f(0.25f, 0.25f); //换算到高一层
                    indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt, nanoflann::SearchParams());

                    pts[i].parent = ret_index[0];                          // parent节点
                    pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor); //在高一层中到parent节点的距离

                    assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
                }
                else //高一层中没有parent节点
                {
                    pts[i].parent = -1;
                    pts[i].parentDist = -1;
                }
            }
        }
        // done.

        for (int i = 0; i < pyrLevelsUsed; i++)
            delete indexes[i];
    }
}
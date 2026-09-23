using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Unity.XR.PICO.TOBSupport;
using Unity.XR.PXR;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class EgoStreamingClientController : MonoBehaviour
{
    private enum EncoderInputMode
    {
        AutoDirectBuffer = 0,
        DirectBuffer = 1,
        JavaByteArray = 2
    }

    private enum CaptureScheduleMode
    {
        LegacyPostWorkDelay = 0,
        PhaseLocked = 1,
        EveryUnityFrame = 2,
        EveryUnityFrameUniqueRateLimited = 3
    }

    private enum DiagnosticPipelineMode
    {
        FullPipeline = 0,
        SchedulerOnly = 1,
        AcquireOnly = 2,
        AcquireAndTracking = 3,
        EncodeAndDiscardOutput = 4
    }

    private enum CameraPollClassification
    {
        FirstNewFrame = 0,
        NewFrame = 1,
        Duplicate = 2,
        NonMonotonic = 3
    }

    private const string PixelFormat = "NV21";
    private const long UnixEpochTicks = 621355968000000000L;
    private const uint PacketMagic = 0x50454731; // "PEG1"
    private const byte PacketVersion = 1;
    private const int PacketHeaderBytes = 20;
    // Absorb one missed source frame without allowing an unbounded catch-up burst after a stall.
    private const double UniquePollingMaxOutputCredits = 2.0;
    // ERROR and SESSION_END must still fit when the normal data queue is saturated.
    private const int TerminationControlPacketReserveCount = 4;
    private const int TerminationControlByteReserve = 64 * 1024;

    private const byte PacketTypeHello = 1;
    private const byte PacketTypeStart = 2;
    private const byte PacketTypeStop = 3;
    private const byte PacketTypeCameraJson = 4;
    private const byte PacketTypeMetadataRow = 5;
    private const byte PacketTypeTimestampRow = 6;
    private const byte PacketTypeHevcSample = 7;
    private const byte PacketTypeSessionEnd = 8;
    private const byte PacketTypeError = 9;
    private const byte PacketTypeMetadataHeader = 10;
    private const byte PacketTypeTimestampHeader = 11;
    private const byte PacketTypeTimeSyncRequest = 12;
    private const byte PacketTypeTimeSyncResponse = 13;

    private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

    [Header("Server")]
    [SerializeField] private string serverHost = "127.0.0.1";
    [SerializeField] private int serverPort = 50051;
    [SerializeField] private float connectTimeoutSeconds = 5f;
    [SerializeField] private int networkSendBufferBytes = 4 * 1024 * 1024;
    [SerializeField] private int maxQueuedPackets = 256;
    [SerializeField] private int maxQueuedBytes = 64 * 1024 * 1024;

    [Header("Capture")]
    [SerializeField] private int targetFps = 45;
    [SerializeField] private float maxCaptureSeconds = 0f;

    [Header("Diagnostics")]
    [SerializeField] private CaptureScheduleMode captureScheduleMode = CaptureScheduleMode.EveryUnityFrameUniqueRateLimited;
    [SerializeField] private DiagnosticPipelineMode diagnosticPipelineMode = DiagnosticPipelineMode.FullPipeline;
    [SerializeField] private bool diagnosticUpdateStatusEveryFrame = false;
    [SerializeField, Min(0.5f)] private float uniquePollingWarmupTimeoutSeconds = 5f;
    [SerializeField, Min(1)] private int uniquePollingMaxConsecutiveAcquireFailures = 90;
    [SerializeField, Min(1)] private int uniquePollingMaxConsecutiveTimestampRegressions = 30;
    [SerializeField, Min(0.25f)] private float uniquePollingNoNewFrameTimeoutSeconds = 1f;

    [Header("HEVC")]
    [SerializeField] private int hevcBitrate = 12 * 1000 * 1000;
    [SerializeField] private int hevcIFrameIntervalSeconds = 1;
    [SerializeField] private bool hevcUseSourceResolution = false;
    [SerializeField] private int hevcWidth = 1280;
    [SerializeField] private int hevcHeight = 960;
    [SerializeField] private EncoderInputMode hevcInputMode = EncoderInputMode.AutoDirectBuffer;

    [Header("Runtime")]
    [SerializeField] private bool autoQuitOnFatalError = false;
    [SerializeField] private bool showStatusUi = true;
    [SerializeField] private float statusLogIntervalSeconds = 0.5f;
    [SerializeField] private float enterpriseBindTimeoutSeconds = 2f;
    [SerializeField] private float eyeTrackingStartTimeoutSeconds = 3f;
    [SerializeField] private bool assumeEyeTrackingAlreadyCalibrated = true;

    private readonly CultureInfo invariantCulture = CultureInfo.InvariantCulture;
    private readonly object sendLock = new object();
    private readonly object commandLock = new object();
    private readonly Queue<OutgoingPacket> sendQueue = new Queue<OutgoingPacket>();

    private TcpClient tcpClient;
    private NetworkStream networkStream;
    private Thread sendThread;
    private Thread receiveThread;
    private volatile bool networkStopRequested;
    private volatile bool connected;
    private volatile bool fatalError;
    private bool cleanedUp;
    private string fatalErrorMessage = string.Empty;
    private int queuedBytes;
    private bool sendPacketInFlight;
    private int networkQueueMaxDepth;
    private int networkQueueMaxBytes;
    private long totalPayloadBytesSent;
    private long totalHevcBytesSent;
    private int packetsSent;
    private int hevcSamplesSent;
    private int sessionEnqueuedPacketCount;
    private int sessionEnqueuedHevcSampleCount;
    private long sessionEnqueuedPayloadBytes;
    private long sessionEnqueuedHevcBytes;

    private bool pendingStart;
    private bool pendingStop;
    private string pendingSessionName = string.Empty;
    private bool captureActive;
    private bool captureStopRequested;
    private bool sessionDataValid;
    private string sessionDataInvalidReason = string.Empty;
    private bool streamFormatReady;

    private RGBCameraParams cameraParameters;
    private EffectiveIntrinsics effectiveIntrinsics;
    private bool cameraOpened;
    private byte[] nv21Buffer;
    private byte[] hevcInputBuffer;
    private HevcEncoderWrapper hevcEncoder;

    private int attemptedFrameCount;
    private int validFrameCount;
    private int invalidFrameCount;
    private int encodedFrameCount;
    private int encoderSampleCount;
    private int metadataRowCount;
    private int timestampRowCount;
    private int firstFrameWidth;
    private int firstFrameHeight;
    private int encoderWidth;
    private int encoderHeight;
    private int nv21FrameByteCount;
    private int hevcInputByteCount;
    private float captureStartTimeSeconds;
    private float captureEndTimeSeconds;
    private float frameIntervalSumMs;
    private int frameIntervalSampleCount;
    private float maxFrameIntervalMs;
    private float acquireSumMs;
    private float copySumMs;
    private float downscaleSumMs;
    private float encodeSumMs;
    private float encoderDequeueInputSumMs;
    private float encoderColorConvertSumMs;
    private float encoderInputPutSumMs;
    private float encoderQueueInputSumMs;
    private float encoderDrainOutputSumMs;
    private float encoderJavaTotalSumMs;
    private float directBufferCreateSumMs;
    private float encodeCallWallSumMs;
    private float codecInputCopyOrConvertSumMs;
    private float jniBridgeSumMs;
    private float networkSendQueueSumMs;
    private int directBufferFrameCount;
    private int javaByteArrayFrameCount;
    private int directFallbackFrameCount;
    private bool directInputDisabledForSession;
    private float gazeSumMs;
    private int validGazeFrameCount;
    private int uniqueCameraFrameCount;
    private int duplicateCameraFrameCount;
    private int nonMonotonicCameraFrameCount;
    private int cameraPollCount;
    private int validCameraPollCount;
    private int invalidCameraPollCount;
    private int observedNewCameraFrameCount;
    private int duplicateCameraPollCount;
    private int nonMonotonicCameraPollCount;
    private int rateLimitedNewCameraFrameCount;
    private int consecutiveDuplicateCameraPollCount;
    private int maxConsecutiveDuplicateCameraPollCount;
    private float maxNoNewCameraFrameDurationMs;
    private int consecutiveAcquireFailureCount;
    private int maxConsecutiveAcquireFailureCount;
    private int consecutiveTimestampRegressionCount;
    private bool hasPreviousPolledCameraFrameTimestamp;
    private ulong previousPolledCameraFrameTimestampNs;
    private bool hasPreviousEmittedCameraFrameTimestamp;
    private ulong previousEmittedCameraFrameTimestampNs;
    private float pollIntervalSumMs;
    private int pollIntervalSampleCount;
    private float maxPollIntervalMs;
    private int pollUnityFrameDeltaSum;
    private int pollUnityFrameDeltaSampleCount;
    private float pollProcessingSumMs;
    private float pollProcessingMaxMs;
    private float warmupAcquireMs;
    private int trackingSampleCount;
    private int schedulerSkippedSlotCount;
    private int schedulerLateCaptureCount;
    private float schedulerLatenessSumMs;
    private float schedulerMaxLatenessMs;
    private int unityFrameDeltaSum;
    private int unityFrameDeltaSampleCount;
    private float unityUnscaledDeltaSumMs;
    private float captureProcessingSumMs;
    private float captureProcessingMaxMs;
    private float streamInitializationMs;
    private int discardedHevcSampleCount;
    private long discardedHevcByteCount;
    private float displayRefreshHz;
    private long encoderSubmittedInputFrameCount;
    private long encoderMatchedOutputFrameCount;
    private long encoderUnknownOutputPtsCount;
    private long encoderDuplicateInputPtsCount;
    private long encoderNonMonotonicInputPtsCount;
    private long encoderPartialOutputSampleCount;
    private int encoderFrameMetaHighWaterMark;
    private int encoderFrameMetaCountAfterEos;
    private bool encoderSawEndOfStream;
    private string encoderFinishError = string.Empty;
    private bool hasPreviousEncoderPresentationTimeUs;
    private long previousEncoderPresentationTimeUs;
    private int encoderPresentationTimestampAdjustedCount;

    private int eyeTrackingWantResult = int.MinValue;
    private int eyeTrackingSupportedResult = int.MinValue;
    private bool eyeTrackingSupported;
    private int eyeTrackingSupportedModesCount;
    private string eyeTrackingSupportedModes = string.Empty;
    private int eyeTrackingStartResult = int.MinValue;
    private int eyeTrackingStopResult = int.MinValue;
    private int eyeTrackingStateResult = int.MinValue;
    private bool eyeTrackingIsTracking;
    private TrackingStateCode eyeTrackingStateCode = TrackingStateCode.PXR_MT_SERVICE_NEED_START;
    private EyeTrackingMode eyeTrackingCurrentMode = EyeTrackingMode.PXR_ETM_NONE;

    private GUIStyle guiStyle;
    private Text statusText;
    private string statusMessage = "Initializing";
    private string lastLoggedStatusMessage = string.Empty;
    private float lastStatusLogTime = -999f;
    private int mainThreadId;

    private struct EffectiveIntrinsics
    {
        public double fx;
        public double fy;
        public double cx;
        public double cy;
    }

    private struct GazeSample
    {
        public bool valid;
        public string source;
        public string failureReason;
        public int motionTrackingDataResult;
        public bool combinedPoseValid;
        public uint combinedStatus;
        public Vector3 eyePosePositionRaw;
        public Vector3 eyePosePositionUnity;
        public Quaternion eyePoseRotationRaw;
        public Quaternion eyePoseRotationUnity;
        public Vector3 gazeWorldDirection;
        public Vector3 legacyEyeDirection;
    }

    private struct FrameRecord
    {
        public int frameIndex;
        public float appTime;
        public ulong frameTimestampNs;
        public long refTimestampUs;
        public long encoderPresentationTimeUs;
        public long acquireStartUnixUs;
        public long acquireEndUnixUs;
        public long acquireMidUnixUs;
        public long xrHeadSampleUnixUs;
        public long gazeSampleUnixUs;
        public int captureResult;
        public int frameStatus;
        public int width;
        public int height;
        public int encoderWidth;
        public int encoderHeight;
        public float acquireMs;
        public float copyMs;
        public float downscaleMs;
        public float encodeMs;
        public float encoderDequeueInputMs;
        public float encoderColorConvertMs;
        public float encoderInputPutMs;
        public float encoderQueueInputMs;
        public float encoderDrainOutputMs;
        public float encoderJavaTotalMs;
        public float directBufferCreateMs;
        public float encodeCallWallMs;
        public float codecInputCopyOrConvertMs;
        public float jniBridgeMs;
        public float networkSendQueueMs;
        public string encoderInputPath;
        public float gazeMs;
        public float frameIntervalMs;
        public string diagnosticPipelineMode;
        public string captureScheduleMode;
        public int unityFrameCount;
        public int unityFramesSincePreviousCapture;
        public float unityUnscaledDeltaMs;
        public float schedulerLatenessMs;
        public int schedulerSkippedSlots;
        public long cameraTimestampDeltaNs;
        public bool cameraFrameIsNew;
        public bool cameraTimestampNonMonotonic;
        public float captureProcessingMs;
        public float streamInitializationMs;
        public int encoderSamplesThisFrame;
        public long encoderBytesThisFrame;
        public int sendQueueDepthBefore;
        public int sendQueueDepthAfterProcessing;
        public int cameraPollIndex;
        public int cameraPollsSincePreviousOutput;
        public int observedNewCameraFramesSincePreviousOutput;
        public int rateLimitedNewCameraFramesSincePreviousOutput;
        public float pollIntervalMs;
        public long observedCameraTimestampDeltaNs;
        public float gazeSampleAppTime;
        public float xrHeadSampleAppTime;
        public bool xrHeadPoseValid;
        public string xrHeadPoseSource;
        public string xrHeadPoseFailureReason;
        public GazeSample gaze;
        public Pose headPose;
        public Pose rgbPose;
        public Pose xrHeadPose;
    }

    private struct EncoderDrainResult
    {
        public float elapsedMs;
        public int sampleCount;
        public long byteCount;
        public List<OutgoingPacket> networkPackets;
    }

    private sealed class OutgoingPacket
    {
        public byte type;
        public string headerJson;
        public byte[] payload;
    }

    private void Start()
    {
        mainThreadId = Thread.CurrentThread.ManagedThreadId;
        EnsureRenderingCameraAvailable();
        StartCoroutine(RunClient());
    }

    private void EnsureRenderingCameraAvailable()
    {
        Camera renderingCamera = Camera.main;
        if (renderingCamera == null || !renderingCamera.isActiveAndEnabled)
        {
            renderingCamera = FindFirstObjectByType<Camera>();
        }

        if (renderingCamera == null)
        {
            Camera[] allCameras = Resources.FindObjectsOfTypeAll<Camera>();
            foreach (Camera candidate in allCameras)
            {
                if (candidate == null || candidate.gameObject == null || !candidate.gameObject.scene.IsValid())
                {
                    continue;
                }

                if (candidate.name == "Main Camera" || candidate.GetComponent<PXR_Manager>() != null)
                {
                    renderingCamera = candidate;
                    break;
                }
            }
        }

        if (renderingCamera == null)
        {
            GameObject cameraObject = new GameObject("EgoStreamingFallbackCamera");
            renderingCamera = cameraObject.AddComponent<Camera>();
            cameraObject.AddComponent<AudioListener>();
            cameraObject.AddComponent<PXR_Manager>();
        }

        if (!renderingCamera.gameObject.activeSelf)
        {
            renderingCamera.gameObject.SetActive(true);
        }

        if (!renderingCamera.enabled)
        {
            renderingCamera.enabled = true;
        }

        try
        {
            renderingCamera.tag = "MainCamera";
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[EgoStreaming] Failed to set rendering camera tag to MainCamera: " + ex.Message);
        }

        renderingCamera.clearFlags = CameraClearFlags.SolidColor;
        renderingCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
        renderingCamera.nearClipPlane = 0.01f;
        renderingCamera.allowHDR = false;
        renderingCamera.allowMSAA = false;
        renderingCamera.stereoTargetEye = StereoTargetEyeMask.Both;

        PXR_Manager manager = renderingCamera.GetComponent<PXR_Manager>();
        if (manager == null)
        {
            manager = renderingCamera.gameObject.AddComponent<PXR_Manager>();
        }

        manager.eyeTracking = true;
        manager.openMRC = true;
    }

    private System.Collections.IEnumerator RunClient()
    {
        PrepareStatusUi();
        EnsurePicoRuntimeFlags();

        if (!TryInitializeEnterpriseService())
        {
            SetFatalError("PXR_Enterprise.InitEnterpriseService(true) failed.");
            Cleanup();
            if (autoQuitOnFatalError)
            {
                QuitApplication();
            }
            yield break;
        }

        yield return BindEnterpriseService();

        if (!TryOpenVstCamera())
        {
            SetFatalError("PXR_Enterprise.OpenVSTCamera() failed.");
            Cleanup();
            if (autoQuitOnFatalError)
            {
                QuitApplication();
            }
            yield break;
        }

        cameraParameters = PXR_Enterprise.GetCameraParameters();
        effectiveIntrinsics = CreateIntrinsics(cameraParameters);
        PXR_Manager.EnableVideoSeeThrough = true;
        yield return InitializeEyeTracking();

        yield return ConnectToServer();
        if (fatalError)
        {
            Cleanup();
            if (autoQuitOnFatalError)
            {
                QuitApplication();
            }
            yield break;
        }

        StartNetworkThreads();
        EnqueueTextPacket(PacketTypeHello, BuildHelloJson(), string.Empty);
        UpdateStatus("Connected. Waiting for START from server.");

        while (!fatalError)
        {
            if (TryConsumeStartCommand(out string sessionName))
            {
                yield return CaptureSession(sessionName);
                UpdateStatus("Session finished. Waiting for START from server.");
            }

            yield return null;
        }

        UpdateStatus("Fatal: " + fatalErrorMessage);
        yield return FlushOutgoingPackets(3f);
        Cleanup();
        if (autoQuitOnFatalError)
        {
            QuitApplication();
        }
    }

    private void EnsurePicoRuntimeFlags()
    {
        PXR_Manager manager = FindFirstObjectByType<PXR_Manager>();
        if (manager == null)
        {
            manager = PXR_Manager.Instance;
        }

        if (manager != null)
        {
            manager.eyeTracking = true;
            manager.openMRC = true;
        }

        PXR_Manager.EnableVideoSeeThrough = true;
    }

    private bool TryInitializeEnterpriseService()
    {
        try
        {
            bool initialized = PXR_Enterprise.InitEnterpriseService(true);
            Debug.Log("[EgoStreaming] InitEnterpriseService: " + initialized);
            return initialized;
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            return false;
        }
    }

    private System.Collections.IEnumerator BindEnterpriseService()
    {
        bool callbackReceived = false;
        try
        {
            PXR_Enterprise.BindEnterpriseService(result =>
            {
                callbackReceived = true;
                Debug.Log("[EgoStreaming] BindEnterpriseService callback: " + result);
            });
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[EgoStreaming] BindEnterpriseService threw: " + ex.Message);
            yield break;
        }

        float deadline = Time.realtimeSinceStartup + Mathf.Max(0f, enterpriseBindTimeoutSeconds);
        while (!callbackReceived && Time.realtimeSinceStartup < deadline)
        {
            UpdateStatus("Binding enterprise service");
            yield return null;
        }
    }

    private bool TryOpenVstCamera()
    {
        try
        {
            cameraOpened = PXR_Enterprise.OpenVSTCamera();
            Debug.Log("[EgoStreaming] OpenVSTCamera: " + cameraOpened);
            return cameraOpened;
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
            return false;
        }
    }

    private System.Collections.IEnumerator InitializeEyeTracking()
    {
        UpdateStatus("Starting eye tracking");
        try
        {
            eyeTrackingWantResult = PXR_MotionTracking.WantEyeTrackingService();
            EyeTrackingMode[] supportedModes = new EyeTrackingMode[8];
            eyeTrackingSupportedResult = PXR_MotionTracking.GetEyeTrackingSupported(ref eyeTrackingSupported, ref eyeTrackingSupportedModesCount, ref supportedModes);
            eyeTrackingSupportedModes = FormatEyeTrackingModes(supportedModes, eyeTrackingSupportedModesCount);

            EyeTrackingStartInfo startInfo = new EyeTrackingStartInfo
            {
                mode = EyeTrackingMode.PXR_ETM_BOTH,
                needCalibration = assumeEyeTrackingAlreadyCalibrated ? (byte)1 : (byte)0
            };
            eyeTrackingStartResult = PXR_MotionTracking.StartEyeTracking(ref startInfo);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[EgoStreaming] Eye tracking init threw: " + ex.Message);
            yield break;
        }

        float deadline = Time.realtimeSinceStartup + Mathf.Max(0f, eyeTrackingStartTimeoutSeconds);
        do
        {
            try
            {
                EyeTrackingState state = default;
                eyeTrackingStateResult = PXR_MotionTracking.GetEyeTrackingState(ref eyeTrackingIsTracking, ref state);
                eyeTrackingStateCode = state.code;
                eyeTrackingCurrentMode = state.currentTrackingMode;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[EgoStreaming] GetEyeTrackingState threw: " + ex.Message);
                yield break;
            }

            if (eyeTrackingIsTracking && eyeTrackingStateCode == TrackingStateCode.PXR_MT_SUCCESS)
            {
                yield break;
            }

            yield return null;
        } while (Time.realtimeSinceStartup < deadline);
    }

    private System.Collections.IEnumerator ConnectToServer()
    {
        UpdateStatus("Connecting to stream server " + serverHost + ":" + serverPort);
        tcpClient = new TcpClient();
        IAsyncResult connectResult = tcpClient.BeginConnect(serverHost, serverPort, null, null);
        float deadline = Time.realtimeSinceStartup + Mathf.Max(0.1f, connectTimeoutSeconds);
        while (!connectResult.IsCompleted && Time.realtimeSinceStartup < deadline)
        {
            yield return null;
        }

        if (!connectResult.IsCompleted)
        {
            SetFatalError("Connection timeout. Is adb reverse tcp:" + serverPort + " tcp:" + serverPort + " active?");
            yield break;
        }

        try
        {
            tcpClient.EndConnect(connectResult);
            tcpClient.NoDelay = true;
            tcpClient.SendBufferSize = Mathf.Max(65536, networkSendBufferBytes);
            networkStream = tcpClient.GetStream();
            connected = true;
        }
        catch (Exception ex)
        {
            SetFatalError("Connect failed: " + ex.Message);
        }
    }

    private void StartNetworkThreads()
    {
        networkStopRequested = false;
        sendThread = new Thread(NetworkSendLoop)
        {
            IsBackground = true,
            Name = "EgoStreamingSend"
        };
        receiveThread = new Thread(NetworkReceiveLoop)
        {
            IsBackground = true,
            Name = "EgoStreamingReceive"
        };
        sendThread.Start();
        receiveThread.Start();
    }

    private bool ShouldAcquireCamera()
    {
        return diagnosticPipelineMode != DiagnosticPipelineMode.SchedulerOnly;
    }

    private bool ShouldSampleTracking()
    {
        return diagnosticPipelineMode == DiagnosticPipelineMode.FullPipeline ||
               diagnosticPipelineMode == DiagnosticPipelineMode.AcquireAndTracking ||
               diagnosticPipelineMode == DiagnosticPipelineMode.EncodeAndDiscardOutput;
    }

    private bool ShouldEncodeFrames()
    {
        return diagnosticPipelineMode == DiagnosticPipelineMode.FullPipeline ||
               diagnosticPipelineMode == DiagnosticPipelineMode.EncodeAndDiscardOutput;
    }

    private bool ShouldSendHevcPayload()
    {
        return diagnosticPipelineMode == DiagnosticPipelineMode.FullPipeline;
    }

    private bool UsesUniquePollingOutputPolicy()
    {
        return captureScheduleMode == CaptureScheduleMode.EveryUnityFrameUniqueRateLimited && ShouldAcquireCamera();
    }

    private System.Collections.IEnumerator CaptureSession(string sessionName)
    {
        captureActive = true;
        streamFormatReady = false;
        ResetCaptureCounters();
        EnqueueTextPacket(PacketTypeHello, BuildSessionStartingJson(sessionName), string.Empty);

        try
        {
            displayRefreshHz = PXR_System.GetSystemDisplayFrequency();
        }
        catch
        {
            displayRefreshHz = 0f;
        }

        bool shouldAcquireCamera = ShouldAcquireCamera();
        bool uniqueRateLimitedPolling = UsesUniquePollingOutputPolicy();
        if (shouldAcquireCamera)
        {
            yield return WarmUpStreamFormat();
        }
        else
        {
            InitializeStreamFormat(0, 0, 0);
        }

        captureStartTimeSeconds = Time.realtimeSinceStartup;
        float interval = 1f / Mathf.Max(1, targetFps);
        float deadline = maxCaptureSeconds > 0f ? captureStartTimeSeconds + maxCaptureSeconds : float.PositiveInfinity;

        if (!fatalError && !captureStopRequested)
        {
            if (uniqueRateLimitedPolling)
            {
                yield return CaptureUniqueRateLimitedLoop(deadline);
            }
            else
            {
                yield return CaptureScheduledLoop(interval, deadline);
            }
        }

        captureEndTimeSeconds = Time.realtimeSinceStartup;
        FinishEncoder();
        ValidateSessionIntegrityBeforeEnd();
        SendSessionEnd(sessionName);
        captureActive = false;
    }

    private System.Collections.IEnumerator WarmUpStreamFormat()
    {
        float deadline = Time.realtimeSinceStartup + Mathf.Max(0.5f, uniquePollingWarmupTimeoutSeconds);
        while (!fatalError && !ShouldStopCaptureSession() && Time.realtimeSinceStartup < deadline)
        {
            FrameRecord warmupRecord = new FrameRecord();
            bool frameValid = AcquireCameraFrame(ref warmupRecord, out Unity.XR.PICO.TOBSupport.Frame frame, out int expectedBytes, false);
            warmupAcquireMs = warmupRecord.acquireMs;
            if (frameValid)
            {
                long initializationStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                InitializeStreamFormat((int)frame.width, (int)frame.height, expectedBytes);
                streamInitializationMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - initializationStartTicks);
                if (streamFormatReady || fatalError)
                {
                    yield break;
                }
            }

            yield return null;
        }

        if (!fatalError && !captureStopRequested && !streamFormatReady)
        {
            SetFatalError("Timed out waiting for a valid VST frame while warming up the stream format.");
        }
    }

    private System.Collections.IEnumerator CaptureScheduledLoop(float interval, float deadline)
    {
        float nextCaptureTime = captureStartTimeSeconds;
        float previousCaptureTime = -1f;
        int previousCaptureUnityFrame = -1;

        while (!fatalError && !ShouldStopCaptureSession() && Time.realtimeSinceStartup < deadline)
        {
            float loopNow = Time.realtimeSinceStartup;
            if (captureScheduleMode != CaptureScheduleMode.EveryUnityFrame && loopNow < nextCaptureTime)
            {
                yield return null;
                continue;
            }

            float schedulerLatenessMs = captureScheduleMode == CaptureScheduleMode.EveryUnityFrame
                ? 0f
                : Mathf.Max(0f, (loopNow - nextCaptureTime) * 1000f);
            int skippedSlots = 0;
            if (captureScheduleMode == CaptureScheduleMode.PhaseLocked ||
                captureScheduleMode == CaptureScheduleMode.EveryUnityFrameUniqueRateLimited)
            {
                int elapsedSlots = Mathf.Max(1, Mathf.FloorToInt((loopNow - nextCaptureTime) / interval) + 1);
                skippedSlots = elapsedSlots - 1;
                nextCaptureTime += elapsedSlots * interval;
                schedulerSkippedSlotCount += skippedSlots;
            }
            else if (captureScheduleMode == CaptureScheduleMode.EveryUnityFrame)
            {
                nextCaptureTime = loopNow;
            }

            RecordSchedulerTiming(schedulerLatenessMs);

            int frameIndex = attemptedFrameCount++;
            float appTime = loopNow;
            float frameIntervalMs = 0f;
            if (previousCaptureTime >= 0f)
            {
                frameIntervalMs = (appTime - previousCaptureTime) * 1000f;
                frameIntervalSumMs += frameIntervalMs;
                frameIntervalSampleCount++;
                maxFrameIntervalMs = Mathf.Max(maxFrameIntervalMs, frameIntervalMs);
            }

            int unityFrameCount = Time.frameCount;
            int unityFramesSincePreviousCapture = previousCaptureUnityFrame >= 0
                ? unityFrameCount - previousCaptureUnityFrame
                : 0;
            if (previousCaptureUnityFrame >= 0)
            {
                unityFrameDeltaSum += unityFramesSincePreviousCapture;
                unityFrameDeltaSampleCount++;
            }
            float unityUnscaledDeltaMs = Time.unscaledDeltaTime * 1000f;
            unityUnscaledDeltaSumMs += unityUnscaledDeltaMs;

            previousCaptureTime = appTime;
            previousCaptureUnityFrame = unityFrameCount;
            if (diagnosticUpdateStatusEveryFrame)
            {
                UpdateStatus("Streaming " + frameIndex + " | queued=" + GetSendQueueDepth());
            }
            CaptureAndStreamOneFrame(
                frameIndex,
                appTime,
                frameIntervalMs,
                unityFrameCount,
                unityFramesSincePreviousCapture,
                unityUnscaledDeltaMs,
                schedulerLatenessMs,
                skippedSlots);

            if (captureScheduleMode == CaptureScheduleMode.LegacyPostWorkDelay)
            {
                nextCaptureTime += interval;
                if (nextCaptureTime < Time.realtimeSinceStartup)
                {
                    schedulerSkippedSlotCount++;
                    nextCaptureTime = Time.realtimeSinceStartup + interval;
                }
            }

            yield return null;
        }
    }

    private System.Collections.IEnumerator CaptureUniqueRateLimitedLoop(float deadline)
    {
        double outputIntervalNs = 1000000000.0 / Math.Max(1, targetFps);
        double outputCredits = 1.0;
        double previousOutputTime = -1.0;
        int previousOutputUnityFrame = -1;
        double previousPollTime = -1.0;
        int previousPollUnityFrame = -1;
        int pollsSincePreviousOutput = 0;
        int observedNewFramesSincePreviousOutput = 0;
        int rateLimitedNewFramesSincePreviousOutput = 0;
        double lastObservedNewFrameTime = Time.realtimeSinceStartupAsDouble;

        while (!fatalError && !ShouldStopCaptureSession() && Time.realtimeSinceStartup < deadline)
        {
            double loopNowDouble = Time.realtimeSinceStartupAsDouble;
            float appTime = (float)loopNowDouble;
            int unityFrameCount = Time.frameCount;
            float unityUnscaledDeltaMs = Time.unscaledDeltaTime * 1000f;
            int pollIndex = cameraPollCount;
            cameraPollCount++;
            pollsSincePreviousOutput++;

            float pollIntervalMs = 0f;
            if (previousPollTime >= 0.0)
            {
                pollIntervalMs = (float)((loopNowDouble - previousPollTime) * 1000.0);
                pollIntervalSumMs += pollIntervalMs;
                pollIntervalSampleCount++;
                maxPollIntervalMs = Mathf.Max(maxPollIntervalMs, pollIntervalMs);
            }

            if (previousPollUnityFrame >= 0)
            {
                pollUnityFrameDeltaSum += unityFrameCount - previousPollUnityFrame;
                pollUnityFrameDeltaSampleCount++;
            }
            previousPollTime = loopNowDouble;
            previousPollUnityFrame = unityFrameCount;

            long pollProcessingStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            FrameRecord record = default;
            record.appTime = appTime;
            record.unityFrameCount = unityFrameCount;
            record.unityUnscaledDeltaMs = unityUnscaledDeltaMs;
            record.cameraPollIndex = pollIndex;
            record.cameraPollsSincePreviousOutput = pollsSincePreviousOutput;
            record.pollIntervalMs = pollIntervalMs;

            // The SDK owns frame.data. Keep acquisition single-threaded and consume accepted data before this iteration yields.
            bool frameValid = AcquireCameraFrame(ref record, out Unity.XR.PICO.TOBSupport.Frame frame, out int expectedBytes, true);
            double observationNow = Time.realtimeSinceStartupAsDouble;
            if (!frameValid)
            {
                invalidCameraPollCount++;
                consecutiveAcquireFailureCount++;
                consecutiveTimestampRegressionCount = 0;
                consecutiveDuplicateCameraPollCount = 0;
                maxConsecutiveAcquireFailureCount = Mathf.Max(maxConsecutiveAcquireFailureCount, consecutiveAcquireFailureCount);
                RecordPollProcessing(pollProcessingStartTicks);
                if (consecutiveAcquireFailureCount >= Mathf.Max(1, uniquePollingMaxConsecutiveAcquireFailures))
                {
                    SetFatalError("VST camera returned " + consecutiveAcquireFailureCount + " consecutive invalid polls.");
                }
                CheckNoNewCameraFrameTimeout(observationNow, lastObservedNewFrameTime);
                yield return null;
                continue;
            }

            validCameraPollCount++;
            consecutiveAcquireFailureCount = 0;
            if (record.width != firstFrameWidth || record.height != firstFrameHeight || expectedBytes != nv21FrameByteCount)
            {
                RecordPollProcessing(pollProcessingStartTicks);
                SetFatalError(
                    "VST frame format changed during capture from " + firstFrameWidth + "x" + firstFrameHeight +
                    " to " + record.width + "x" + record.height + ". Restart the session before continuing.");
                yield break;
            }

            CameraPollClassification classification = ClassifyPolledCameraTimestamp(frame.timestamp, out long observedTimestampDeltaNs);
            record.observedCameraTimestampDeltaNs = observedTimestampDeltaNs;
            if (classification == CameraPollClassification.Duplicate)
            {
                consecutiveTimestampRegressionCount = 0;
                duplicateCameraPollCount++;
                consecutiveDuplicateCameraPollCount++;
                maxConsecutiveDuplicateCameraPollCount = Mathf.Max(
                    maxConsecutiveDuplicateCameraPollCount,
                    consecutiveDuplicateCameraPollCount);
                CheckNoNewCameraFrameTimeout(observationNow, lastObservedNewFrameTime);
                RecordPollProcessing(pollProcessingStartTicks);
                yield return null;
                continue;
            }

            if (classification == CameraPollClassification.NonMonotonic)
            {
                nonMonotonicCameraPollCount++;
                consecutiveDuplicateCameraPollCount = 0;
                consecutiveTimestampRegressionCount++;
                RecordPollProcessing(pollProcessingStartTicks);
                if (consecutiveTimestampRegressionCount >= Mathf.Max(1, uniquePollingMaxConsecutiveTimestampRegressions))
                {
                    SetFatalError(
                        "VST camera timestamp regressed for " + consecutiveTimestampRegressionCount +
                        " consecutive polls; the camera clock likely restarted.");
                }
                CheckNoNewCameraFrameTimeout(observationNow, lastObservedNewFrameTime);
                yield return null;
                continue;
            }

            consecutiveTimestampRegressionCount = 0;
            consecutiveDuplicateCameraPollCount = 0;
            lastObservedNewFrameTime = observationNow;
            observedNewCameraFrameCount++;
            observedNewFramesSincePreviousOutput++;
            if (classification == CameraPollClassification.NewFrame)
            {
                outputCredits = Math.Min(
                    UniquePollingMaxOutputCredits,
                    outputCredits + observedTimestampDeltaNs / outputIntervalNs);
            }

            if (outputCredits + 0.000000001 < 1.0)
            {
                rateLimitedNewCameraFrameCount++;
                rateLimitedNewFramesSincePreviousOutput++;
                RecordPollProcessing(pollProcessingStartTicks);
                yield return null;
                continue;
            }

            outputCredits = Math.Max(0.0, outputCredits - 1.0);
            int skippedOutputSlots = 0;
            float schedulerLatenessMs = 0f;
            RecordSchedulerTiming(schedulerLatenessMs);

            int frameIndex = attemptedFrameCount++;
            float frameIntervalMs = 0f;
            if (previousOutputTime >= 0.0)
            {
                frameIntervalMs = (float)((loopNowDouble - previousOutputTime) * 1000.0);
                frameIntervalSumMs += frameIntervalMs;
                frameIntervalSampleCount++;
                maxFrameIntervalMs = Mathf.Max(maxFrameIntervalMs, frameIntervalMs);
            }

            int unityFramesSincePreviousOutput = previousOutputUnityFrame >= 0
                ? unityFrameCount - previousOutputUnityFrame
                : 0;
            if (previousOutputUnityFrame >= 0)
            {
                unityFrameDeltaSum += unityFramesSincePreviousOutput;
                unityFrameDeltaSampleCount++;
            }
            unityUnscaledDeltaSumMs += unityUnscaledDeltaMs;

            record.frameIndex = frameIndex;
            record.frameIntervalMs = frameIntervalMs;
            record.unityFramesSincePreviousCapture = unityFramesSincePreviousOutput;
            record.schedulerLatenessMs = schedulerLatenessMs;
            record.schedulerSkippedSlots = skippedOutputSlots;
            record.cameraFrameIsNew = true;
            record.cameraTimestampDeltaNs = hasPreviousEmittedCameraFrameTimestamp
                ? (long)(frame.timestamp - previousEmittedCameraFrameTimestampNs)
                : 0L;
            record.cameraPollsSincePreviousOutput = pollsSincePreviousOutput;
            record.observedNewCameraFramesSincePreviousOutput = observedNewFramesSincePreviousOutput;
            record.rateLimitedNewCameraFramesSincePreviousOutput = rateLimitedNewFramesSincePreviousOutput;
            record.diagnosticPipelineMode = diagnosticPipelineMode.ToString();
            record.captureScheduleMode = captureScheduleMode.ToString();
            record.sendQueueDepthBefore = GetSendQueueDepth();
            record.gaze = CreateEmptyGaze("not_sampled");

            bool completed = ProcessAcquiredFrame(record, frame, expectedBytes, true, pollProcessingStartTicks);
            RecordPollProcessing(pollProcessingStartTicks);
            if (completed)
            {
                uniqueCameraFrameCount++;
                previousEmittedCameraFrameTimestampNs = frame.timestamp;
                hasPreviousEmittedCameraFrameTimestamp = true;
                previousOutputTime = loopNowDouble;
                previousOutputUnityFrame = unityFrameCount;
                pollsSincePreviousOutput = 0;
                observedNewFramesSincePreviousOutput = 0;
                rateLimitedNewFramesSincePreviousOutput = 0;
            }

            if (diagnosticUpdateStatusEveryFrame)
            {
                UpdateStatus("Polling " + cameraPollCount + " | output=" + validFrameCount + " | queued=" + GetSendQueueDepth());
            }

            yield return null;
        }
    }

    private void CheckNoNewCameraFrameTimeout(double currentTime, double lastObservedNewFrameTime)
    {
        float noNewFrameDurationMs = (float)((currentTime - lastObservedNewFrameTime) * 1000.0);
        maxNoNewCameraFrameDurationMs = Mathf.Max(maxNoNewCameraFrameDurationMs, noNewFrameDurationMs);
        if (noNewFrameDurationMs >= Mathf.Max(0.25f, uniquePollingNoNewFrameTimeoutSeconds) * 1000f)
        {
            SetFatalError(
                "VST camera has not produced a new monotonic timestamp for " +
                noNewFrameDurationMs.ToString("F1", invariantCulture) + " ms.");
        }
    }

    private FrameRecord CreateFrameRecord(
        int frameIndex,
        float appTime,
        float frameIntervalMs,
        int unityFrameCount,
        int unityFramesSincePreviousCapture,
        float unityUnscaledDeltaMs,
        float schedulerLatenessMs,
        int schedulerSkippedSlots)
    {
        return new FrameRecord
        {
            frameIndex = frameIndex,
            appTime = appTime,
            frameIntervalMs = frameIntervalMs,
            diagnosticPipelineMode = diagnosticPipelineMode.ToString(),
            captureScheduleMode = captureScheduleMode.ToString(),
            unityFrameCount = unityFrameCount,
            unityFramesSincePreviousCapture = unityFramesSincePreviousCapture,
            unityUnscaledDeltaMs = unityUnscaledDeltaMs,
            schedulerLatenessMs = schedulerLatenessMs,
            schedulerSkippedSlots = schedulerSkippedSlots,
            sendQueueDepthBefore = GetSendQueueDepth(),
            gaze = CreateEmptyGaze("not_sampled")
        };
    }

    private void CaptureAndStreamOneFrame(
        int frameIndex,
        float appTime,
        float frameIntervalMs,
        int unityFrameCount,
        int unityFramesSincePreviousCapture,
        float unityUnscaledDeltaMs,
        float schedulerLatenessMs,
        int schedulerSkippedSlots)
    {
        long captureProcessingStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        FrameRecord record = CreateFrameRecord(
            frameIndex,
            appTime,
            frameIntervalMs,
            unityFrameCount,
            unityFramesSincePreviousCapture,
            unityUnscaledDeltaMs,
            schedulerLatenessMs,
            schedulerSkippedSlots);

        if (!ShouldAcquireCamera())
        {
            long schedulerTimestampUs = UnixTimeMicrosecondsNow();
            record.acquireStartUnixUs = schedulerTimestampUs;
            record.acquireEndUnixUs = schedulerTimestampUs;
            record.acquireMidUnixUs = schedulerTimestampUs;
            record.refTimestampUs = schedulerTimestampUs;
            record.captureResult = -1;
            record.frameStatus = -1;
            record.encoderInputPath = "scheduler_only";
            CompleteFrameRecord(record, captureProcessingStartTicks);
            return;
        }

        int pollIndex = cameraPollCount++;
        record.cameraPollIndex = pollIndex;
        record.cameraPollsSincePreviousOutput = 1;
        record.pollIntervalMs = frameIntervalMs;
        if (frameIntervalSampleCount > 0)
        {
            pollIntervalSumMs += frameIntervalMs;
            pollIntervalSampleCount++;
            maxPollIntervalMs = Mathf.Max(maxPollIntervalMs, frameIntervalMs);
        }
        if (unityFramesSincePreviousCapture > 0)
        {
            pollUnityFrameDeltaSum += unityFramesSincePreviousCapture;
            pollUnityFrameDeltaSampleCount++;
        }

        bool frameValid = AcquireCameraFrame(ref record, out Unity.XR.PICO.TOBSupport.Frame frame, out int expectedBytes, true);
        if (frameValid)
        {
            validCameraPollCount++;
            CameraPollClassification classification = ClassifyPolledCameraTimestamp(frame.timestamp, out long observedTimestampDeltaNs);
            record.observedCameraTimestampDeltaNs = observedTimestampDeltaNs;
            record.cameraTimestampDeltaNs = observedTimestampDeltaNs;
            if (classification == CameraPollClassification.FirstNewFrame || classification == CameraPollClassification.NewFrame)
            {
                observedNewCameraFrameCount++;
                uniqueCameraFrameCount++;
                record.cameraFrameIsNew = true;
                record.observedNewCameraFramesSincePreviousOutput = 1;
            }
            else if (classification == CameraPollClassification.Duplicate)
            {
                duplicateCameraPollCount++;
                duplicateCameraFrameCount++;
            }
            else
            {
                nonMonotonicCameraPollCount++;
                nonMonotonicCameraFrameCount++;
                record.cameraTimestampNonMonotonic = true;
            }
        }
        else
        {
            invalidCameraPollCount++;
        }

        ProcessAcquiredFrame(record, frame, expectedBytes, frameValid, captureProcessingStartTicks);
        RecordPollProcessing(captureProcessingStartTicks);
    }

    private bool AcquireCameraFrame(
        ref FrameRecord record,
        out Unity.XR.PICO.TOBSupport.Frame frame,
        out int expectedBytes,
        bool includeInStatistics)
    {
        record.acquireStartUnixUs = UnixTimeMicrosecondsNow();
        long startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        int result = PXR_Enterprise.AcquireVSTCameraFrame(out frame);
        record.acquireEndUnixUs = UnixTimeMicrosecondsNow();
        record.acquireMidUnixUs = record.acquireStartUnixUs + ((record.acquireEndUnixUs - record.acquireStartUnixUs) / 2L);
        record.refTimestampUs = record.acquireMidUnixUs;
        record.acquireMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
        if (includeInStatistics)
        {
            acquireSumMs += record.acquireMs;
        }

        int width = (int)frame.width;
        int height = (int)frame.height;
        expectedBytes = 0;
        bool frameValid = result == 0 && width > 0 && height > 0 && frame.data != IntPtr.Zero;
        if (frameValid)
        {
            long calculatedBytes = (long)width * height * 3L / 2L;
            if ((width & 1) != 0 || (height & 1) != 0 || calculatedBytes <= 0L || calculatedBytes > int.MaxValue)
            {
                frameValid = false;
            }
            else
            {
                expectedBytes = (int)calculatedBytes;
                if (frame.datasize < (uint)expectedBytes)
                {
                    frameValid = false;
                }
            }
        }

        record.frameTimestampNs = frame.timestamp;
        record.captureResult = result;
        record.frameStatus = frame.status;
        record.width = width;
        record.height = height;
        return frameValid;
    }

    private CameraPollClassification ClassifyPolledCameraTimestamp(ulong timestampNs, out long deltaNs)
    {
        if (!hasPreviousPolledCameraFrameTimestamp)
        {
            hasPreviousPolledCameraFrameTimestamp = true;
            previousPolledCameraFrameTimestampNs = timestampNs;
            deltaNs = 0L;
            return CameraPollClassification.FirstNewFrame;
        }

        if (timestampNs > previousPolledCameraFrameTimestampNs)
        {
            deltaNs = (long)(timestampNs - previousPolledCameraFrameTimestampNs);
            previousPolledCameraFrameTimestampNs = timestampNs;
            return CameraPollClassification.NewFrame;
        }

        if (timestampNs == previousPolledCameraFrameTimestampNs)
        {
            deltaNs = 0L;
            return CameraPollClassification.Duplicate;
        }

        deltaNs = -(long)(previousPolledCameraFrameTimestampNs - timestampNs);
        return CameraPollClassification.NonMonotonic;
    }

    private bool ProcessAcquiredFrame(
        FrameRecord record,
        Unity.XR.PICO.TOBSupport.Frame frame,
        int expectedBytes,
        bool frameValid,
        long captureProcessingStartTicks)
    {
        long startTicks;
        if (frameValid)
        {
            record.headPose = ToUnityPose(frame.pose);
            record.rgbPose = ToRgbCameraPose(cameraParameters, frame.pose);
        }

        if (ShouldSampleTracking())
        {
            trackingSampleCount++;
            record.xrHeadSampleUnixUs = UnixTimeMicrosecondsNow();
            record.xrHeadSampleAppTime = Time.realtimeSinceStartup;
            record.xrHeadPoseValid = TrySampleXrHeadPose(out record.xrHeadPose, out record.xrHeadPoseSource, out record.xrHeadPoseFailureReason);
            record.gazeSampleUnixUs = UnixTimeMicrosecondsNow();
            record.gazeSampleAppTime = Time.realtimeSinceStartup;
            startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            record.gaze = SampleRawGaze();
            record.gazeMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
            gazeSumMs += record.gazeMs;
            if (record.gaze.valid)
            {
                validGazeFrameCount++;
            }
        }
        else
        {
            record.xrHeadPoseFailureReason = "diagnostic_tracking_disabled";
            record.gaze = CreateEmptyGaze("diagnostic_tracking_disabled");
        }

        if (frameValid && !streamFormatReady)
        {
            long initializationStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            InitializeStreamFormat(record.width, record.height, expectedBytes);
            record.streamInitializationMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - initializationStartTicks);
            streamInitializationMs = record.streamInitializationMs;
            if (fatalError || !streamFormatReady)
            {
                return false;
            }
        }
        else if (frameValid && (record.width != firstFrameWidth || record.height != firstFrameHeight || expectedBytes != nv21FrameByteCount))
        {
            frameValid = false;
        }

        if (!frameValid)
        {
            invalidFrameCount++;
            return CompleteFrameRecord(record, captureProcessingStartTicks);
        }

        if (!ShouldEncodeFrames())
        {
            record.encoderInputPath = "not_encoded";
            validFrameCount++;
            return CompleteFrameRecord(record, captureProcessingStartTicks);
        }

        bool canUseDirectInput = CanUseDirectEncoderInput(record.width, record.height);
        byte[] encoderInput = null;
        if (!canUseDirectInput)
        {
            startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            EnsureNv21Buffer(expectedBytes);
            Marshal.Copy(frame.data, nv21Buffer, 0, expectedBytes);
            record.copyMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
            copySumMs += record.copyMs;
            encoderInput = nv21Buffer;
        }

        if (encoderWidth != record.width || encoderHeight != record.height)
        {
            if (encoderInput == null)
            {
                startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                EnsureNv21Buffer(expectedBytes);
                Marshal.Copy(frame.data, nv21Buffer, 0, expectedBytes);
                record.copyMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
                copySumMs += record.copyMs;
                encoderInput = nv21Buffer;
            }

            startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            EnsureHevcInputBuffer(hevcInputByteCount);
            DownscaleNv21Nearest(nv21Buffer, record.width, record.height, hevcInputBuffer, encoderWidth, encoderHeight);
            record.downscaleMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
            downscaleSumMs += record.downscaleMs;
            encoderInput = hevcInputBuffer;
        }

        record.encoderWidth = encoderWidth;
        record.encoderHeight = encoderHeight;
        long presentationTimeUs = CreateMonotonicEncoderPresentationTimeUs(record.refTimestampUs);
        record.encoderPresentationTimeUs = presentationTimeUs;
        long encodeStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        long encodeCallStartTicks = encodeStartTicks;
        bool encoded;
        if (canUseDirectInput)
        {
            encoded = hevcEncoder.EncodeFrameDirect(frame.data, expectedBytes, presentationTimeUs, record.frameIndex, record.refTimestampUs);
            record.encoderInputPath = hevcEncoder.LastInputPath;
            if (!encoded &&
                hevcInputMode == EncoderInputMode.AutoDirectBuffer &&
                hevcEncoder.LastDirectFailureSafeForFallback)
            {
                directFallbackFrameCount++;
                directInputDisabledForSession = true;
                startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                EnsureNv21Buffer(expectedBytes);
                Marshal.Copy(frame.data, nv21Buffer, 0, expectedBytes);
                record.copyMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
                copySumMs += record.copyMs;
                encodeCallStartTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                encoded = hevcEncoder.EncodeFrame(nv21Buffer, presentationTimeUs, record.frameIndex, record.refTimestampUs);
                record.encoderInputPath = hevcEncoder.LastInputPath;
            }
        }
        else
        {
            encoded = hevcEncoder.EncodeFrame(encoderInput, presentationTimeUs, record.frameIndex, record.refTimestampUs);
            record.encoderInputPath = hevcEncoder.LastInputPath;
        }

        record.encodeCallWallMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - encodeCallStartTicks);
        if (!encoded)
        {
            SetFatalError("HEVC encode failed: " + hevcEncoder.LastError);
            return false;
        }

        EncoderDrainResult drainResult = DrainEncoderSamples();
        record.networkSendQueueMs = drainResult.elapsedMs;
        record.encoderSamplesThisFrame = drainResult.sampleCount;
        record.encoderBytesThisFrame = drainResult.byteCount;
        record.encodeMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - encodeStartTicks);
        record.encoderDequeueInputMs = hevcEncoder.LastDequeueInputMs;
        record.encoderColorConvertMs = hevcEncoder.LastColorConvertMs;
        record.encoderInputPutMs = hevcEncoder.LastInputPutMs;
        record.encoderQueueInputMs = hevcEncoder.LastQueueInputMs;
        record.encoderDrainOutputMs = hevcEncoder.LastDrainOutputMs;
        record.encoderJavaTotalMs = hevcEncoder.LastJavaTotalMs;
        record.directBufferCreateMs = hevcEncoder.LastDirectBufferCreateMs;
        record.codecInputCopyOrConvertMs = record.encoderColorConvertMs + record.encoderInputPutMs;
        record.jniBridgeMs = Mathf.Max(0f, record.encodeCallWallMs - record.encoderJavaTotalMs - record.directBufferCreateMs);
        encodeSumMs += record.encodeMs;
        encoderDequeueInputSumMs += record.encoderDequeueInputMs;
        encoderColorConvertSumMs += record.encoderColorConvertMs;
        encoderInputPutSumMs += record.encoderInputPutMs;
        encoderQueueInputSumMs += record.encoderQueueInputMs;
        encoderDrainOutputSumMs += record.encoderDrainOutputMs;
        encoderJavaTotalSumMs += record.encoderJavaTotalMs;
        directBufferCreateSumMs += record.directBufferCreateMs;
        encodeCallWallSumMs += record.encodeCallWallMs;
        codecInputCopyOrConvertSumMs += record.codecInputCopyOrConvertMs;
        jniBridgeSumMs += record.jniBridgeMs;
        networkSendQueueSumMs += record.networkSendQueueMs;
        if (record.encoderInputPath == "direct_byte_buffer")
        {
            directBufferFrameCount++;
        }
        else
        {
            javaByteArrayFrameCount++;
        }
        encodedFrameCount++;
        validFrameCount++;
        return CompleteFrameRecord(record, captureProcessingStartTicks, drainResult.networkPackets);
    }

    private long CreateMonotonicEncoderPresentationTimeUs(long preferredTimestampUs)
    {
        long presentationTimeUs = preferredTimestampUs;
        if (hasPreviousEncoderPresentationTimeUs && presentationTimeUs <= previousEncoderPresentationTimeUs)
        {
            presentationTimeUs = previousEncoderPresentationTimeUs + 1L;
            encoderPresentationTimestampAdjustedCount++;
        }
        hasPreviousEncoderPresentationTimeUs = true;
        previousEncoderPresentationTimeUs = presentationTimeUs;
        return presentationTimeUs;
    }

    private void RecordSchedulerTiming(float schedulerLatenessMs)
    {
        if (schedulerLatenessMs > 0.25f)
        {
            schedulerLateCaptureCount++;
        }
        schedulerLatenessSumMs += schedulerLatenessMs;
        schedulerMaxLatenessMs = Mathf.Max(schedulerMaxLatenessMs, schedulerLatenessMs);
    }

    private void RecordPollProcessing(long pollProcessingStartTicks)
    {
        float elapsedMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - pollProcessingStartTicks);
        pollProcessingSumMs += elapsedMs;
        pollProcessingMaxMs = Mathf.Max(pollProcessingMaxMs, elapsedMs);
    }

    private bool CompleteFrameRecord(
        FrameRecord record,
        long captureProcessingStartTicks,
        List<OutgoingPacket> hevcPackets = null)
    {
        record.captureProcessingMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - captureProcessingStartTicks);
        captureProcessingSumMs += record.captureProcessingMs;
        captureProcessingMaxMs = Mathf.Max(captureProcessingMaxMs, record.captureProcessingMs);
        record.sendQueueDepthAfterProcessing = GetSendQueueDepth();
        if (!TryEnqueueFramePacketBatch(
                hevcPackets,
                PacketTypeMetadataRow,
                BuildMetadataRow(record) + "\n",
                PacketTypeTimestampRow,
                BuildTimestampRow(record) + "\n"))
        {
            SetFatalError("Network queue backed up while atomically enqueueing HEVC/metadata/timestamp data.");
            return false;
        }

        metadataRowCount++;
        timestampRowCount++;
        return true;
    }

    private void InitializeStreamFormat(int width, int height, int expectedBytes)
    {
        firstFrameWidth = width;
        firstFrameHeight = height;
        nv21FrameByteCount = expectedBytes;
        effectiveIntrinsics = CreateIntrinsics(cameraParameters);
        string encoderConfigJson = string.Empty;

        if (ShouldEncodeFrames())
        {
            ResolveEncoderSize(width, height, out encoderWidth, out encoderHeight);
            hevcInputByteCount = encoderWidth * encoderHeight * 3 / 2;
            if (hevcInputMode == EncoderInputMode.DirectBuffer && (encoderWidth != width || encoderHeight != height))
            {
                SetFatalError("DirectBuffer encoder input requires source resolution. Enable HEVC Use Source Resolution or switch HEVC Input Mode to AutoDirectBuffer/JavaByteArray.");
                return;
            }

            hevcEncoder = new HevcEncoderWrapper();
            if (!hevcEncoder.Start(encoderWidth, encoderHeight, hevcBitrate, Mathf.Max(1, targetFps), Mathf.Max(1, hevcIFrameIntervalSeconds)))
            {
                SetFatalError("HEVC encoder start failed: " + hevcEncoder.LastError);
                return;
            }

            encoderConfigJson = hevcEncoder.ConfigJson;
        }
        else
        {
            encoderWidth = 0;
            encoderHeight = 0;
            hevcInputByteCount = 0;
        }

        streamFormatReady = true;
        EnqueueTextPacket(PacketTypeCameraJson, "{}", BuildCameraJson(width, height, encoderWidth, encoderHeight, encoderConfigJson));
        EnqueueTextPacket(PacketTypeMetadataHeader, "{}", MetadataHeader() + "\n");
        EnqueueTextPacket(PacketTypeTimestampHeader, "{}", TimestampHeader() + "\n");
    }

    private bool CanUseDirectEncoderInput(int sourceWidth, int sourceHeight)
    {
        if (hevcInputMode == EncoderInputMode.JavaByteArray)
        {
            return false;
        }

        if (directInputDisabledForSession)
        {
            return false;
        }

        return encoderWidth == sourceWidth && encoderHeight == sourceHeight;
    }

    private void FinishEncoder()
    {
        if (hevcEncoder == null)
        {
            return;
        }

        hevcEncoder.Finish(UnixTimeMicrosecondsNow());
        EncoderDrainResult finalDrainResult = DrainEncoderSamples();
        if (finalDrainResult.networkPackets != null &&
            finalDrainResult.networkPackets.Count > 0 &&
            !TryEnqueuePacketBatch(finalDrainResult.networkPackets))
        {
            SetFatalError("Network queue backed up while enqueueing final HEVC samples.");
        }
        encoderSubmittedInputFrameCount = hevcEncoder.SubmittedInputFrameCount;
        encoderMatchedOutputFrameCount = hevcEncoder.MatchedOutputFrameCount;
        encoderUnknownOutputPtsCount = hevcEncoder.UnknownOutputPtsCount;
        encoderDuplicateInputPtsCount = hevcEncoder.DuplicateInputPtsCount;
        encoderNonMonotonicInputPtsCount = hevcEncoder.NonMonotonicInputPtsCount;
        encoderPartialOutputSampleCount = hevcEncoder.PartialOutputSampleCount;
        encoderFrameMetaHighWaterMark = hevcEncoder.FrameMetaHighWaterMark;
        encoderFrameMetaCountAfterEos = hevcEncoder.FrameMetaCountAfterEos;
        encoderSawEndOfStream = hevcEncoder.SawEndOfStream;
        encoderFinishError = hevcEncoder.LastError ?? string.Empty;
        hevcEncoder.Dispose();
        hevcEncoder = null;

        if (!IsEncoderLifecycleHealthy())
        {
            SetFatalError(
                "HEVC finalization integrity check failed: submitted=" + encoderSubmittedInputFrameCount +
                ", matched=" + encoderMatchedOutputFrameCount +
                ", unknown_pts=" + encoderUnknownOutputPtsCount +
                ", pending_meta=" + encoderFrameMetaCountAfterEos +
                ", saw_eos=" + encoderSawEndOfStream +
                (string.IsNullOrEmpty(encoderFinishError) ? string.Empty : ", error=" + encoderFinishError));
        }
    }

    private bool IsEncoderLifecycleHealthy()
    {
        return !ShouldEncodeFrames() ||
            (string.IsNullOrEmpty(encoderFinishError) &&
             encoderSubmittedInputFrameCount == encodedFrameCount &&
             encoderMatchedOutputFrameCount == encoderSubmittedInputFrameCount &&
             encoderUnknownOutputPtsCount == 0L &&
             encoderDuplicateInputPtsCount == 0L &&
             encoderNonMonotonicInputPtsCount == 0L &&
             encoderFrameMetaCountAfterEos == 0 &&
             encoderSawEndOfStream);
    }

    private void ValidateSessionIntegrityBeforeEnd()
    {
        if (!streamFormatReady)
        {
            SetFatalError("Session ended before the stream format and output headers were initialized.");
            return;
        }

        if (!IsEncoderLifecycleHealthy())
        {
            SetFatalError("HEVC lifecycle integrity check failed before session end.");
            return;
        }

        if (metadataRowCount != timestampRowCount || attemptedFrameCount != metadataRowCount)
        {
            SetFatalError(
                "Output row integrity check failed: attempted=" + attemptedFrameCount +
                ", metadata=" + metadataRowCount +
                ", timestamps=" + timestampRowCount + ".");
            return;
        }

        if (ShouldSendHevcPayload() && sessionEnqueuedHevcSampleCount != encoderSampleCount)
        {
            SetFatalError(
                "HEVC transport integrity check failed: generated_samples=" + encoderSampleCount +
                ", enqueued_samples=" + sessionEnqueuedHevcSampleCount + ".");
        }
    }

    private EncoderDrainResult DrainEncoderSamples()
    {
        EncoderDrainResult result = default;
        if (hevcEncoder == null)
        {
            return result;
        }

        long startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        while (hevcEncoder.PendingSampleCount > 0)
        {
            string headerJson = ShouldSendHevcPayload() ? hevcEncoder.PeekSampleHeaderJson() : string.Empty;
            byte[] data = hevcEncoder.PopSampleData();
            if (data == null || data.Length == 0)
            {
                continue;
            }

            encoderSampleCount++;
            result.sampleCount++;
            result.byteCount += data.Length;
            if (ShouldSendHevcPayload())
            {
                if (result.networkPackets == null)
                {
                    result.networkPackets = new List<OutgoingPacket>(2);
                }
                result.networkPackets.Add(new OutgoingPacket
                {
                    type = PacketTypeHevcSample,
                    headerJson = headerJson,
                    payload = data
                });
            }
            else
            {
                discardedHevcSampleCount++;
                discardedHevcByteCount += data.Length;
            }
        }

        result.elapsedMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
        return result;
    }

    private void SendSessionEnd(string sessionName)
    {
        float duration = Mathf.Max(0.0001f, captureEndTimeSeconds - captureStartTimeSeconds);
        bool outputRowsAligned = metadataRowCount == timestampRowCount && attemptedFrameCount == metadataRowCount;
        bool encoderMappingHealthy = IsEncoderLifecycleHealthy();
        bool hevcTransportAligned = !ShouldSendHevcPayload() || sessionEnqueuedHevcSampleCount == encoderSampleCount;
        bool finalSessionDataValid = sessionDataValid && outputRowsAligned && encoderMappingHealthy && hevcTransportAligned;
        string outputPolicy = UsesUniquePollingOutputPolicy()
            ? "unique_camera_timestamp_token_bucket"
            : "scheduled_attempt";

        StringBuilder json = new StringBuilder(3072);
        json.Append('{');
        json.Append("\"session_name\":").Append(JsonString(sessionName)).Append(',');
        json.Append("\"metadata_schema_version\":2,");
        json.Append("\"session_data_valid\":").Append(JsonBool(finalSessionDataValid)).Append(',');
        json.Append("\"session_data_invalid_reason\":").Append(JsonString(sessionDataInvalidReason)).Append(',');
        json.Append("\"capture_fatal_error\":").Append(JsonBool(fatalError)).Append(',');
        json.Append("\"capture_fatal_error_message\":").Append(JsonString(fatalErrorMessage)).Append(',');
        json.Append("\"encoder_finish_error\":").Append(JsonString(encoderFinishError)).Append(',');
        json.Append("\"frame_index_semantics\":").Append(JsonString("continuous_output_sequence")).Append(',');
        json.Append("\"capture_output_policy\":").Append(JsonString(outputPolicy)).Append(',');
        json.Append("\"target_fps\":").Append(targetFps).Append(',');
        json.Append("\"capture_schedule_mode\":").Append(JsonString(captureScheduleMode.ToString())).Append(',');
        json.Append("\"diagnostic_pipeline_mode\":").Append(JsonString(diagnosticPipelineMode.ToString())).Append(',');
        json.Append("\"diagnostic_update_status_every_frame\":").Append(JsonBool(diagnosticUpdateStatusEveryFrame)).Append(',');
        json.Append("\"display_refresh_hz\":").Append(JsonNumber(displayRefreshHz)).Append(',');
        json.Append("\"attempted_frame_count\":").Append(attemptedFrameCount).Append(',');
        json.Append("\"output_frame_count\":").Append(metadataRowCount).Append(',');
        json.Append("\"metadata_row_count\":").Append(metadataRowCount).Append(',');
        json.Append("\"timestamp_row_count\":").Append(timestampRowCount).Append(',');
        json.Append("\"output_rows_aligned\":").Append(JsonBool(outputRowsAligned)).Append(',');
        json.Append("\"hevc_transport_aligned\":").Append(JsonBool(hevcTransportAligned)).Append(',');
        json.Append("\"valid_frame_count\":").Append(validFrameCount).Append(',');
        json.Append("\"invalid_frame_count\":").Append(invalidFrameCount).Append(',');
        json.Append("\"unique_camera_frame_count\":").Append(uniqueCameraFrameCount).Append(',');
        json.Append("\"emitted_unique_camera_frame_count\":").Append(uniqueCameraFrameCount).Append(',');
        json.Append("\"duplicate_camera_frame_count\":").Append(duplicateCameraFrameCount).Append(',');
        json.Append("\"non_monotonic_camera_frame_count\":").Append(nonMonotonicCameraFrameCount).Append(',');
        json.Append("\"camera_poll_count\":").Append(cameraPollCount).Append(',');
        json.Append("\"valid_camera_poll_count\":").Append(validCameraPollCount).Append(',');
        json.Append("\"invalid_camera_poll_count\":").Append(invalidCameraPollCount).Append(',');
        json.Append("\"observed_new_camera_frame_count\":").Append(observedNewCameraFrameCount).Append(',');
        json.Append("\"duplicate_camera_poll_count\":").Append(duplicateCameraPollCount).Append(',');
        json.Append("\"non_monotonic_camera_poll_count\":").Append(nonMonotonicCameraPollCount).Append(',');
        json.Append("\"rate_limited_new_camera_frame_count\":").Append(rateLimitedNewCameraFrameCount).Append(',');
        json.Append("\"max_consecutive_duplicate_camera_poll_count\":").Append(maxConsecutiveDuplicateCameraPollCount).Append(',');
        json.Append("\"max_no_new_camera_frame_duration_ms\":").Append(JsonNumber(maxNoNewCameraFrameDurationMs)).Append(',');
        json.Append("\"max_consecutive_acquire_failure_count\":").Append(maxConsecutiveAcquireFailureCount).Append(',');
        json.Append("\"max_allowed_consecutive_acquire_failures\":").Append(uniquePollingMaxConsecutiveAcquireFailures).Append(',');
        json.Append("\"max_allowed_consecutive_timestamp_regressions\":").Append(uniquePollingMaxConsecutiveTimestampRegressions).Append(',');
        json.Append("\"encoded_frame_count\":").Append(encodedFrameCount).Append(',');
        json.Append("\"hevc_sample_count\":").Append(encoderSampleCount).Append(',');
        json.Append("\"discarded_hevc_sample_count\":").Append(discardedHevcSampleCount).Append(',');
        json.Append("\"discarded_hevc_byte_count\":").Append(discardedHevcByteCount).Append(',');
        json.Append("\"encoder_submitted_input_frame_count\":").Append(encoderSubmittedInputFrameCount).Append(',');
        json.Append("\"encoder_matched_output_frame_count\":").Append(encoderMatchedOutputFrameCount).Append(',');
        json.Append("\"encoder_unknown_output_pts_count\":").Append(encoderUnknownOutputPtsCount).Append(',');
        json.Append("\"encoder_duplicate_input_pts_count\":").Append(encoderDuplicateInputPtsCount).Append(',');
        json.Append("\"encoder_non_monotonic_input_pts_count\":").Append(encoderNonMonotonicInputPtsCount).Append(',');
        json.Append("\"encoder_partial_output_sample_count\":").Append(encoderPartialOutputSampleCount).Append(',');
        json.Append("\"encoder_frame_meta_high_water_mark\":").Append(encoderFrameMetaHighWaterMark).Append(',');
        json.Append("\"encoder_frame_meta_count_after_eos\":").Append(encoderFrameMetaCountAfterEos).Append(',');
        json.Append("\"encoder_saw_end_of_stream\":").Append(JsonBool(encoderSawEndOfStream)).Append(',');
        json.Append("\"encoder_mapping_healthy\":").Append(JsonBool(encoderMappingHealthy)).Append(',');
        json.Append("\"encoder_presentation_timestamp_adjusted_count\":").Append(encoderPresentationTimestampAdjustedCount).Append(',');
        json.Append("\"source_width\":").Append(firstFrameWidth).Append(',');
        json.Append("\"source_height\":").Append(firstFrameHeight).Append(',');
        json.Append("\"encoder_width\":").Append(encoderWidth).Append(',');
        json.Append("\"encoder_height\":").Append(encoderHeight).Append(',');
        json.Append("\"hevc_input_byte_count\":").Append(hevcInputByteCount).Append(',');
        json.Append("\"encoder_input_mode\":").Append(JsonString(hevcInputMode.ToString())).Append(',');
        json.Append("\"direct_input_disabled_for_session\":").Append(JsonBool(directInputDisabledForSession)).Append(',');
        json.Append("\"direct_buffer_frame_count\":").Append(directBufferFrameCount).Append(',');
        json.Append("\"java_byte_array_frame_count\":").Append(javaByteArrayFrameCount).Append(',');
        json.Append("\"direct_fallback_frame_count\":").Append(directFallbackFrameCount).Append(',');
        json.Append("\"actual_capture_hz\":").Append(JsonNumber(attemptedFrameCount / duration)).Append(',');
        json.Append("\"actual_output_hz\":").Append(JsonNumber(metadataRowCount / duration)).Append(',');
        json.Append("\"unique_camera_frame_hz\":").Append(JsonNumber(uniqueCameraFrameCount / duration)).Append(',');
        json.Append("\"emitted_unique_camera_frame_hz\":").Append(JsonNumber(uniqueCameraFrameCount / duration)).Append(',');
        json.Append("\"camera_poll_hz\":").Append(JsonNumber(cameraPollCount / duration)).Append(',');
        json.Append("\"valid_camera_poll_hz\":").Append(JsonNumber(validCameraPollCount / duration)).Append(',');
        json.Append("\"observed_new_camera_frame_hz\":").Append(JsonNumber(observedNewCameraFrameCount / duration)).Append(',');
        json.Append("\"average_frame_interval_ms\":").Append(JsonNumber(frameIntervalSampleCount > 0 ? frameIntervalSumMs / frameIntervalSampleCount : 0f)).Append(',');
        json.Append("\"max_frame_interval_ms\":").Append(JsonNumber(maxFrameIntervalMs)).Append(',');
        json.Append("\"average_poll_interval_ms\":").Append(JsonNumber(pollIntervalSampleCount > 0 ? pollIntervalSumMs / pollIntervalSampleCount : 0f)).Append(',');
        json.Append("\"max_poll_interval_ms\":").Append(JsonNumber(maxPollIntervalMs)).Append(',');
        json.Append("\"average_unity_frames_between_polls\":").Append(JsonNumber(pollUnityFrameDeltaSampleCount > 0 ? (float)pollUnityFrameDeltaSum / pollUnityFrameDeltaSampleCount : 0f)).Append(',');
        json.Append("\"average_poll_processing_ms\":").Append(JsonNumber(cameraPollCount > 0 ? pollProcessingSumMs / cameraPollCount : 0f)).Append(',');
        json.Append("\"max_poll_processing_ms\":").Append(JsonNumber(pollProcessingMaxMs)).Append(',');
        json.Append("\"warmup_acquire_ms\":").Append(JsonNumber(warmupAcquireMs)).Append(',');
        json.Append("\"scheduler_late_capture_count\":").Append(schedulerLateCaptureCount).Append(',');
        json.Append("\"scheduler_skipped_slot_count\":").Append(schedulerSkippedSlotCount).Append(',');
        json.Append("\"average_scheduler_lateness_ms\":").Append(JsonNumber(attemptedFrameCount > 0 ? schedulerLatenessSumMs / attemptedFrameCount : 0f)).Append(',');
        json.Append("\"max_scheduler_lateness_ms\":").Append(JsonNumber(schedulerMaxLatenessMs)).Append(',');
        json.Append("\"average_unity_frames_between_captures\":").Append(JsonNumber(unityFrameDeltaSampleCount > 0 ? (float)unityFrameDeltaSum / unityFrameDeltaSampleCount : 0f)).Append(',');
        json.Append("\"average_unity_unscaled_delta_ms\":").Append(JsonNumber(attemptedFrameCount > 0 ? unityUnscaledDeltaSumMs / attemptedFrameCount : 0f)).Append(',');
        json.Append("\"average_capture_processing_ms\":").Append(JsonNumber(metadataRowCount > 0 ? captureProcessingSumMs / metadataRowCount : 0f)).Append(',');
        json.Append("\"max_capture_processing_ms\":").Append(JsonNumber(captureProcessingMaxMs)).Append(',');
        json.Append("\"stream_initialization_ms\":").Append(JsonNumber(streamInitializationMs)).Append(',');
        json.Append("\"average_acquire_ms\":").Append(JsonNumber(cameraPollCount > 0 ? acquireSumMs / cameraPollCount : 0f)).Append(',');
        json.Append("\"average_copy_ms\":").Append(JsonNumber(validFrameCount > 0 ? copySumMs / validFrameCount : 0f)).Append(',');
        json.Append("\"average_downscale_ms\":").Append(JsonNumber(validFrameCount > 0 ? downscaleSumMs / validFrameCount : 0f)).Append(',');
        json.Append("\"average_encode_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encodeSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_dequeue_input_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderDequeueInputSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_color_convert_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderColorConvertSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_input_put_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderInputPutSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_queue_input_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderQueueInputSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_drain_output_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderDrainOutputSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encoder_java_total_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encoderJavaTotalSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_direct_buffer_create_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? directBufferCreateSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_encode_call_wall_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? encodeCallWallSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_codec_input_copy_or_convert_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? codecInputCopyOrConvertSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_jni_bridge_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? jniBridgeSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"average_network_send_queue_ms\":").Append(JsonNumber(encodedFrameCount > 0 ? networkSendQueueSumMs / encodedFrameCount : 0f)).Append(',');
        json.Append("\"tracking_sample_count\":").Append(trackingSampleCount).Append(',');
        json.Append("\"average_gaze_ms\":").Append(JsonNumber(trackingSampleCount > 0 ? gazeSumMs / trackingSampleCount : 0f)).Append(',');
        json.Append("\"valid_gaze_frame_count\":").Append(validGazeFrameCount).Append(',');
        json.Append("\"network_queue_max_depth\":").Append(networkQueueMaxDepth).Append(',');
        json.Append("\"network_queue_max_bytes\":").Append(networkQueueMaxBytes).Append(',');
        json.Append("\"network_termination_control_packet_reserve_count\":").Append(TerminationControlPacketReserveCount).Append(',');
        json.Append("\"network_termination_control_byte_reserve\":").Append(TerminationControlByteReserve).Append(',');
        json.Append("\"session_enqueued_packet_count\":").Append(sessionEnqueuedPacketCount).Append(',');
        json.Append("\"session_enqueued_hevc_sample_count\":").Append(sessionEnqueuedHevcSampleCount).Append(',');
        json.Append("\"session_enqueued_payload_bytes\":").Append(sessionEnqueuedPayloadBytes).Append(',');
        json.Append("\"session_enqueued_hevc_bytes\":").Append(sessionEnqueuedHevcBytes).Append(',');
        json.Append("\"total_sent_counter_scope\":").Append(JsonString("connection_cumulative_snapshot_before_session_end_transmit")).Append(',');
        json.Append("\"total_payload_bytes_sent\":").Append(totalPayloadBytesSent).Append(',');
        json.Append("\"total_hevc_bytes_sent\":").Append(totalHevcBytesSent);
        json.Append('}');
        EnqueueTextPacket(PacketTypeSessionEnd, json.ToString(), string.Empty);
    }

    private void ResetCaptureCounters()
    {
        attemptedFrameCount = 0;
        validFrameCount = 0;
        invalidFrameCount = 0;
        encodedFrameCount = 0;
        encoderSampleCount = 0;
        metadataRowCount = 0;
        timestampRowCount = 0;
        firstFrameWidth = 0;
        firstFrameHeight = 0;
        encoderWidth = 0;
        encoderHeight = 0;
        nv21FrameByteCount = 0;
        hevcInputByteCount = 0;
        frameIntervalSumMs = 0f;
        frameIntervalSampleCount = 0;
        maxFrameIntervalMs = 0f;
        acquireSumMs = 0f;
        copySumMs = 0f;
        downscaleSumMs = 0f;
        encodeSumMs = 0f;
        encoderDequeueInputSumMs = 0f;
        encoderColorConvertSumMs = 0f;
        encoderInputPutSumMs = 0f;
        encoderQueueInputSumMs = 0f;
        encoderDrainOutputSumMs = 0f;
        encoderJavaTotalSumMs = 0f;
        directBufferCreateSumMs = 0f;
        encodeCallWallSumMs = 0f;
        codecInputCopyOrConvertSumMs = 0f;
        jniBridgeSumMs = 0f;
        networkSendQueueSumMs = 0f;
        directBufferFrameCount = 0;
        javaByteArrayFrameCount = 0;
        directFallbackFrameCount = 0;
        directInputDisabledForSession = false;
        gazeSumMs = 0f;
        validGazeFrameCount = 0;
        trackingSampleCount = 0;
        uniqueCameraFrameCount = 0;
        duplicateCameraFrameCount = 0;
        nonMonotonicCameraFrameCount = 0;
        cameraPollCount = 0;
        validCameraPollCount = 0;
        invalidCameraPollCount = 0;
        observedNewCameraFrameCount = 0;
        duplicateCameraPollCount = 0;
        nonMonotonicCameraPollCount = 0;
        rateLimitedNewCameraFrameCount = 0;
        consecutiveDuplicateCameraPollCount = 0;
        maxConsecutiveDuplicateCameraPollCount = 0;
        maxNoNewCameraFrameDurationMs = 0f;
        consecutiveAcquireFailureCount = 0;
        maxConsecutiveAcquireFailureCount = 0;
        consecutiveTimestampRegressionCount = 0;
        hasPreviousPolledCameraFrameTimestamp = false;
        previousPolledCameraFrameTimestampNs = 0UL;
        hasPreviousEmittedCameraFrameTimestamp = false;
        previousEmittedCameraFrameTimestampNs = 0UL;
        pollIntervalSumMs = 0f;
        pollIntervalSampleCount = 0;
        maxPollIntervalMs = 0f;
        pollUnityFrameDeltaSum = 0;
        pollUnityFrameDeltaSampleCount = 0;
        pollProcessingSumMs = 0f;
        pollProcessingMaxMs = 0f;
        warmupAcquireMs = 0f;
        schedulerSkippedSlotCount = 0;
        schedulerLateCaptureCount = 0;
        schedulerLatenessSumMs = 0f;
        schedulerMaxLatenessMs = 0f;
        unityFrameDeltaSum = 0;
        unityFrameDeltaSampleCount = 0;
        unityUnscaledDeltaSumMs = 0f;
        captureProcessingSumMs = 0f;
        captureProcessingMaxMs = 0f;
        streamInitializationMs = 0f;
        discardedHevcSampleCount = 0;
        discardedHevcByteCount = 0L;
        encoderSubmittedInputFrameCount = 0L;
        encoderMatchedOutputFrameCount = 0L;
        encoderUnknownOutputPtsCount = 0L;
        encoderDuplicateInputPtsCount = 0L;
        encoderNonMonotonicInputPtsCount = 0L;
        encoderPartialOutputSampleCount = 0L;
        encoderFrameMetaHighWaterMark = 0;
        encoderFrameMetaCountAfterEos = 0;
        encoderSawEndOfStream = false;
        hasPreviousEncoderPresentationTimeUs = false;
        previousEncoderPresentationTimeUs = 0L;
        encoderPresentationTimestampAdjustedCount = 0;
        displayRefreshHz = 0f;
        captureStopRequested = false;
        sessionDataValid = true;
        sessionDataInvalidReason = string.Empty;
        encoderFinishError = string.Empty;
        lock (sendLock)
        {
            networkQueueMaxDepth = sendQueue.Count;
            networkQueueMaxBytes = queuedBytes;
            sessionEnqueuedPacketCount = 0;
            sessionEnqueuedHevcSampleCount = 0;
            sessionEnqueuedPayloadBytes = 0L;
            sessionEnqueuedHevcBytes = 0L;
        }
        streamFormatReady = false;
    }

    private void EnsureNv21Buffer(int requiredBytes)
    {
        if (nv21Buffer == null || nv21Buffer.Length != requiredBytes)
        {
            nv21Buffer = new byte[requiredBytes];
        }
    }

    private void EnsureHevcInputBuffer(int requiredBytes)
    {
        if (hevcInputBuffer == null || hevcInputBuffer.Length != requiredBytes)
        {
            hevcInputBuffer = new byte[requiredBytes];
        }
    }

    private void ResolveEncoderSize(int sourceWidth, int sourceHeight, out int resolvedWidth, out int resolvedHeight)
    {
        if (hevcUseSourceResolution || hevcWidth <= 0 || hevcHeight <= 0)
        {
            resolvedWidth = MakeEven(sourceWidth);
            resolvedHeight = MakeEven(sourceHeight);
            return;
        }

        resolvedWidth = Mathf.Min(MakeEven(hevcWidth), MakeEven(sourceWidth));
        resolvedHeight = Mathf.Min(MakeEven(hevcHeight), MakeEven(sourceHeight));
        resolvedWidth = Mathf.Max(2, resolvedWidth);
        resolvedHeight = Mathf.Max(2, resolvedHeight);
    }

    private static int MakeEven(int value)
    {
        return Mathf.Max(2, value & ~1);
    }

    private static void DownscaleNv21Nearest(byte[] source, int sourceWidth, int sourceHeight, byte[] destination, int destinationWidth, int destinationHeight)
    {
        int sourceYSize = sourceWidth * sourceHeight;
        int destinationYSize = destinationWidth * destinationHeight;

        for (int y = 0; y < destinationHeight; y++)
        {
            int sourceY = y * sourceHeight / destinationHeight;
            int sourceRow = sourceY * sourceWidth;
            int destinationRow = y * destinationWidth;
            for (int x = 0; x < destinationWidth; x++)
            {
                int sourceX = x * sourceWidth / destinationWidth;
                destination[destinationRow + x] = source[sourceRow + sourceX];
            }
        }

        int sourceChromaHeight = sourceHeight / 2;
        int destinationChromaHeight = destinationHeight / 2;
        int sourceChromaWidth = sourceWidth / 2;
        int destinationChromaWidth = destinationWidth / 2;
        for (int y = 0; y < destinationChromaHeight; y++)
        {
            int sourceY = y * sourceChromaHeight / destinationChromaHeight;
            int sourceRow = sourceY * sourceWidth;
            int destinationRow = y * destinationWidth;
            for (int x = 0; x < destinationChromaWidth; x++)
            {
                int sourceX = x * sourceChromaWidth / destinationChromaWidth;
                int sourceIndex = sourceYSize + sourceRow + sourceX * 2;
                int destinationIndex = destinationYSize + destinationRow + x * 2;
                destination[destinationIndex] = source[sourceIndex];
                destination[destinationIndex + 1] = source[sourceIndex + 1];
            }
        }
    }

    private GazeSample SampleRawGaze()
    {
        GazeSample sample = CreateEmptyGaze("not_sampled");
        if (TrySampleMotionTrackingGaze(ref sample))
        {
            return sample;
        }

        TrySampleLegacyInputGaze(ref sample);
        return sample;
    }

    private bool TrySampleMotionTrackingGaze(ref GazeSample sample)
    {
        sample.source = "motion_tracking";

        try
        {
            EyeTrackingDataGetInfo getInfo = new EyeTrackingDataGetInfo
            {
                displayTime = 0,
                flags = EyeTrackingDataGetFlags.PXR_EYE_POSITION | EyeTrackingDataGetFlags.PXR_EYE_ORIENTATION
            };
            EyeTrackingData data = new EyeTrackingData
            {
                eyeDatas = new PerEyeData[(int)PerEyeUsage.EyeCount]
            };

            int dataResult = PXR_MotionTracking.GetEyeTrackingData(ref getInfo, ref data);
            sample.motionTrackingDataResult = dataResult;
            if (dataResult != 0)
            {
                sample.failureReason = "motion_tracking_data_result_" + dataResult;
                return false;
            }

            PerEyeData combined = data.eyeDatas[(int)PerEyeUsage.Combined];
            sample.combinedPoseValid = combined.isPoseValid != 0;
            sample.combinedStatus = sample.combinedPoseValid ? 1u : 0u;
            if (!sample.combinedPoseValid)
            {
                sample.failureReason = "motion_tracking_combined_pose_invalid";
                return false;
            }

            sample.eyePosePositionRaw = ToVector3(combined.pose.position);
            sample.eyePosePositionUnity = ToUnityPosition(sample.eyePosePositionRaw);
            sample.eyePoseRotationRaw = ToQuaternion(combined.pose.orientation);
            sample.eyePoseRotationUnity = ToUnityRotation(sample.eyePoseRotationRaw);
            sample.gazeWorldDirection = sample.eyePoseRotationUnity * Vector3.forward;
            sample.legacyEyeDirection = sample.gazeWorldDirection;
            sample.valid = sample.gazeWorldDirection.sqrMagnitude > 0.000001f;
            sample.failureReason = sample.valid ? string.Empty : "motion_tracking_zero_direction";
            return sample.valid;
        }
        catch (Exception ex)
        {
            sample.failureReason = "motion_tracking_exception_" + ex.GetType().Name;
            return false;
        }
    }

    private bool TrySampleLegacyInputGaze(ref GazeSample sample)
    {
        sample.source = "legacy_input_device";

        bool statusOk = PXR_EyeTracking.GetCombinedEyePoseStatus(out sample.combinedStatus);
        bool originOk = PXR_EyeTracking.GetCombineEyeGazePoint(out Vector3 legacyOrigin);
        bool directionOk = PXR_EyeTracking.GetCombineEyeGazeVector(out Vector3 legacyDirection);
        sample.eyePosePositionRaw = legacyOrigin;
        sample.eyePosePositionUnity = legacyOrigin;
        sample.eyePoseRotationRaw = Quaternion.identity;
        sample.eyePoseRotationUnity = Quaternion.identity;
        sample.gazeWorldDirection = legacyDirection;
        sample.legacyEyeDirection = legacyDirection;
        sample.combinedPoseValid = statusOk && sample.combinedStatus == 1;
        sample.valid = statusOk && originOk && directionOk && sample.combinedPoseValid && sample.gazeWorldDirection.sqrMagnitude > 0.000001f;
        sample.failureReason = sample.valid ? string.Empty : "legacy_unavailable";
        return sample.valid;
    }

    private bool TrySampleXrHeadPose(out Pose pose, out string source, out string failureReason)
    {
        if (TrySampleXrNodePose(XRNode.CenterEye, out pose, out failureReason))
        {
            source = "xr_center_eye";
            return true;
        }
        if (TrySampleXrNodePose(XRNode.Head, out pose, out failureReason))
        {
            source = "xr_head";
            return true;
        }

        source = "none";
        return false;
    }

    private static bool TrySampleXrNodePose(XRNode node, out Pose pose, out string failureReason)
    {
        pose = Pose.identity;
        failureReason = string.Empty;
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);
        if (!device.isValid)
        {
            failureReason = node + "_device_invalid";
            return false;
        }

        bool posOk = device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position);
        bool rotOk = device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation);
        if (!posOk || !rotOk)
        {
            failureReason = node + "_pose_unavailable";
            return false;
        }

        pose = new Pose(position, rotation);
        return true;
    }

    private bool TryEnqueuePacket(byte type, string headerJson, byte[] payload)
    {
        if (payload == null)
        {
            payload = Array.Empty<byte>();
        }
        if (headerJson == null)
        {
            headerJson = "{}";
        }

        byte[] headerBytes = Utf8NoBom.GetBytes(headerJson);
        long packetBytes = (long)headerBytes.Length + payload.Length + PacketHeaderBytes;
        bool terminationControlPacket = type == PacketTypeError || type == PacketTypeSessionEnd;
        long packetLimit = Math.Max(1, maxQueuedPackets) +
            (terminationControlPacket ? TerminationControlPacketReserveCount : 0L);
        long byteLimit = Math.Min(
            int.MaxValue,
            Math.Max(PacketHeaderBytes, maxQueuedBytes) +
            (terminationControlPacket ? TerminationControlByteReserve : 0L));
        lock (sendLock)
        {
            if (sendQueue.Count >= packetLimit || (long)queuedBytes + packetBytes > byteLimit || packetBytes > int.MaxValue)
            {
                return false;
            }

            sendQueue.Enqueue(new OutgoingPacket
            {
                type = type,
                headerJson = headerJson,
                payload = payload
            });
            RecordSessionEnqueuedPacketLocked(type, payload.Length);
            queuedBytes += (int)packetBytes;
            if (sendQueue.Count > networkQueueMaxDepth)
            {
                networkQueueMaxDepth = sendQueue.Count;
            }
            if (queuedBytes > networkQueueMaxBytes)
            {
                networkQueueMaxBytes = queuedBytes;
            }
            Monitor.Pulse(sendLock);
            return true;
        }
    }

    private bool TryEnqueuePacketBatch(List<OutgoingPacket> packets)
    {
        if (packets == null || packets.Count == 0)
        {
            return true;
        }

        long batchBytes = 0L;
        for (int i = 0; i < packets.Count; i++)
        {
            OutgoingPacket packet = packets[i];
            if (packet == null)
            {
                return false;
            }
            packet.headerJson = packet.headerJson ?? "{}";
            packet.payload = packet.payload ?? Array.Empty<byte>();
            batchBytes += Utf8NoBom.GetByteCount(packet.headerJson) + packet.payload.Length + PacketHeaderBytes;
        }
        if (batchBytes > int.MaxValue)
        {
            return false;
        }

        lock (sendLock)
        {
            if (sendQueue.Count + packets.Count > maxQueuedPackets || (long)queuedBytes + batchBytes > maxQueuedBytes)
            {
                return false;
            }

            for (int i = 0; i < packets.Count; i++)
            {
                OutgoingPacket packet = packets[i];
                sendQueue.Enqueue(packet);
                RecordSessionEnqueuedPacketLocked(packet.type, packet.payload.Length);
            }
            queuedBytes += (int)batchBytes;
            networkQueueMaxDepth = Mathf.Max(networkQueueMaxDepth, sendQueue.Count);
            networkQueueMaxBytes = Mathf.Max(networkQueueMaxBytes, queuedBytes);
            Monitor.Pulse(sendLock);
            return true;
        }
    }

    private bool TryEnqueueFramePacketBatch(
        List<OutgoingPacket> leadingPackets,
        byte firstType,
        string firstText,
        byte secondType,
        string secondText)
    {
        const string headerJson = "{}";
        byte[] firstPayload = string.IsNullOrEmpty(firstText) ? Array.Empty<byte>() : Utf8NoBom.GetBytes(firstText);
        byte[] secondPayload = string.IsNullOrEmpty(secondText) ? Array.Empty<byte>() : Utf8NoBom.GetBytes(secondText);
        int leadingPacketCount = leadingPackets == null ? 0 : leadingPackets.Count;
        long batchBytes =
            (long)Utf8NoBom.GetByteCount(headerJson) * 2L +
            firstPayload.Length +
            secondPayload.Length +
            PacketHeaderBytes * 2L;
        for (int i = 0; i < leadingPacketCount; i++)
        {
            OutgoingPacket packet = leadingPackets[i];
            if (packet == null)
            {
                return false;
            }
            packet.headerJson = packet.headerJson ?? "{}";
            packet.payload = packet.payload ?? Array.Empty<byte>();
            batchBytes += Utf8NoBom.GetByteCount(packet.headerJson) + packet.payload.Length + PacketHeaderBytes;
        }
        if (batchBytes > int.MaxValue)
        {
            return false;
        }

        lock (sendLock)
        {
            int packetCount = leadingPacketCount + 2;
            if (sendQueue.Count + packetCount > maxQueuedPackets || (long)queuedBytes + batchBytes > maxQueuedBytes)
            {
                return false;
            }

            for (int i = 0; i < leadingPacketCount; i++)
            {
                OutgoingPacket packet = leadingPackets[i];
                sendQueue.Enqueue(packet);
                RecordSessionEnqueuedPacketLocked(packet.type, packet.payload.Length);
            }
            sendQueue.Enqueue(new OutgoingPacket
            {
                type = firstType,
                headerJson = headerJson,
                payload = firstPayload
            });
            RecordSessionEnqueuedPacketLocked(firstType, firstPayload.Length);
            sendQueue.Enqueue(new OutgoingPacket
            {
                type = secondType,
                headerJson = headerJson,
                payload = secondPayload
            });
            RecordSessionEnqueuedPacketLocked(secondType, secondPayload.Length);
            queuedBytes += (int)batchBytes;
            networkQueueMaxDepth = Mathf.Max(networkQueueMaxDepth, sendQueue.Count);
            networkQueueMaxBytes = Mathf.Max(networkQueueMaxBytes, queuedBytes);
            Monitor.Pulse(sendLock);
            return true;
        }
    }

    private void RecordSessionEnqueuedPacketLocked(byte type, int payloadBytes)
    {
        if (!captureActive)
        {
            return;
        }

        sessionEnqueuedPacketCount++;
        sessionEnqueuedPayloadBytes += payloadBytes;
        if (type == PacketTypeHevcSample)
        {
            sessionEnqueuedHevcSampleCount++;
            sessionEnqueuedHevcBytes += payloadBytes;
        }
    }

    private void EnqueueTextPacket(byte type, string headerJson, string textPayload)
    {
        byte[] payload = string.IsNullOrEmpty(textPayload) ? Array.Empty<byte>() : Utf8NoBom.GetBytes(textPayload);
        if (!TryEnqueuePacket(type, headerJson, payload))
        {
            SetFatalError("Network queue backed up while enqueueing packet type " + type);
        }
    }

    private void NetworkSendLoop()
    {
        try
        {
            while (true)
            {
                OutgoingPacket packet = null;
                int packetBytes = 0;
                lock (sendLock)
                {
                    while (sendQueue.Count == 0 && !networkStopRequested)
                    {
                        Monitor.Wait(sendLock, 20);
                    }
                    if (sendQueue.Count == 0 && networkStopRequested)
                    {
                        break;
                    }
                    packet = sendQueue.Dequeue();
                    int headerLength = Utf8NoBom.GetByteCount(packet.headerJson ?? "{}");
                    int payloadLength = packet.payload == null ? 0 : packet.payload.Length;
                    packetBytes = PacketHeaderBytes + headerLength + payloadLength;
                    queuedBytes = Mathf.Max(0, queuedBytes - packetBytes);
                    sendPacketInFlight = true;
                }

                try
                {
                    WritePacket(networkStream, packet.type, packet.headerJson, packet.payload);
                    packetsSent++;
                    int payloadBytes = packet.payload == null ? 0 : packet.payload.Length;
                    totalPayloadBytesSent += payloadBytes;
                    if (packet.type == PacketTypeHevcSample)
                    {
                        hevcSamplesSent++;
                        totalHevcBytesSent += payloadBytes;
                    }
                }
                finally
                {
                    lock (sendLock)
                    {
                        sendPacketInFlight = false;
                        Monitor.PulseAll(sendLock);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (!networkStopRequested)
            {
                SetFatalError("Network send failed: " + ex.Message);
            }
        }
    }

    private System.Collections.IEnumerator FlushOutgoingPackets(float timeoutSeconds)
    {
        double deadline = Time.realtimeSinceStartupAsDouble + Math.Max(0.1, timeoutSeconds);
        while (Time.realtimeSinceStartupAsDouble < deadline)
        {
            bool drained;
            lock (sendLock)
            {
                drained = sendQueue.Count == 0 && !sendPacketInFlight;
            }

            if (drained)
            {
                yield break;
            }

            yield return null;
        }

        int remainingPackets;
        bool packetInFlight;
        lock (sendLock)
        {
            remainingPackets = sendQueue.Count;
            packetInFlight = sendPacketInFlight;
        }
        Debug.LogWarning(
            "[EgoStreaming] Timed out flushing the network queue before cleanup: queued=" +
            remainingPackets + ", in_flight=" + packetInFlight + ".");
    }

    private void NetworkReceiveLoop()
    {
        try
        {
            while (!networkStopRequested)
            {
                IncomingPacket packet = ReadPacket(networkStream);
                long packetReceivedUnixUs = UnixTimeMicrosecondsNow();
                if (packet.type == PacketTypeStart)
                {
                    lock (commandLock)
                    {
                        pendingSessionName = ExtractJsonString(packet.headerJson, "session_name", "pico_session");
                        pendingStart = true;
                        pendingStop = false;
                    }
                }
                else if (packet.type == PacketTypeStop)
                {
                    lock (commandLock)
                    {
                        pendingStop = true;
                    }
                }
                else if (packet.type == PacketTypeTimeSyncRequest)
                {
                    HandleTimeSyncRequest(packet.headerJson, packetReceivedUnixUs);
                }
            }
        }
        catch (Exception ex)
        {
            if (!networkStopRequested)
            {
                SetFatalError("Network receive failed: " + ex.Message);
            }
        }
    }

    private bool TryConsumeStartCommand(out string sessionName)
    {
        lock (commandLock)
        {
            if (pendingStart && !captureActive)
            {
                sessionName = pendingSessionName;
                pendingStart = false;
                return true;
            }
        }

        sessionName = string.Empty;
        return false;
    }

    private bool TryConsumeStopCommand()
    {
        lock (commandLock)
        {
            if (pendingStop)
            {
                pendingStop = false;
                return true;
            }
        }

        return false;
    }

    private bool ShouldStopCaptureSession()
    {
        if (!captureStopRequested && TryConsumeStopCommand())
        {
            captureStopRequested = true;
        }

        return captureStopRequested;
    }

    private void HandleTimeSyncRequest(string headerJson, long clientRecvUnixUs)
    {
        string calibrationId = ExtractJsonString(headerJson, "calibration_id", string.Empty);
        long seq = ExtractJsonLong(headerJson, "seq", -1L);
        long serverSendUnixUs = ExtractJsonLong(headerJson, "server_send_unix_us", 0L);
        if (string.IsNullOrEmpty(calibrationId) || seq < 0L || serverSendUnixUs <= 0L)
        {
            return;
        }

        long clientSendUnixUs = UnixTimeMicrosecondsNow();
        StringBuilder json = new StringBuilder(256);
        json.Append('{');
        json.Append("\"calibration_id\":").Append(JsonString(calibrationId)).Append(',');
        json.Append("\"seq\":").Append(seq.ToString(invariantCulture)).Append(',');
        json.Append("\"server_send_unix_us\":").Append(serverSendUnixUs.ToString(invariantCulture)).Append(',');
        json.Append("\"client_recv_unix_us\":").Append(clientRecvUnixUs.ToString(invariantCulture)).Append(',');
        json.Append("\"client_send_unix_us\":").Append(clientSendUnixUs.ToString(invariantCulture));
        json.Append('}');
        if (!TryEnqueuePacket(PacketTypeTimeSyncResponse, json.ToString(), Array.Empty<byte>()))
        {
            SetFatalError("Network queue backed up while enqueueing time sync response");
        }
    }

    private int GetSendQueueDepth()
    {
        lock (sendLock)
        {
            return sendQueue.Count;
        }
    }

    private void SetFatalError(string message)
    {
        if (captureActive && sessionDataValid)
        {
            sessionDataValid = false;
            sessionDataInvalidReason = message ?? string.Empty;
        }

        if (fatalError)
        {
            return;
        }

        fatalErrorMessage = message;
        fatalError = true;
        if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
        {
            UpdateStatus("Fatal: " + message);
            Debug.LogError("[EgoStreaming] " + message);
        }
        else
        {
            statusMessage = "Fatal: " + message;
        }
        try
        {
            if (connected)
            {
                TryEnqueuePacket(PacketTypeError, "{\"message\":" + JsonString(message) + "}", Array.Empty<byte>());
            }
        }
        catch
        {
            // Best-effort error reporting.
        }
    }

    private static void WritePacket(NetworkStream stream, byte type, string headerJson, byte[] payload)
    {
        if (stream == null)
        {
            throw new IOException("Network stream is null.");
        }
        if (headerJson == null)
        {
            headerJson = "{}";
        }
        if (payload == null)
        {
            payload = Array.Empty<byte>();
        }

        byte[] headerBytes = Utf8NoBom.GetBytes(headerJson);
        byte[] packetHeader = new byte[PacketHeaderBytes];
        WriteUInt32BE(packetHeader, 0, PacketMagic);
        packetHeader[4] = PacketVersion;
        packetHeader[5] = type;
        WriteUInt16BE(packetHeader, 6, 0);
        WriteUInt32BE(packetHeader, 8, (uint)headerBytes.Length);
        WriteUInt64BE(packetHeader, 12, (ulong)payload.Length);
        stream.Write(packetHeader, 0, packetHeader.Length);
        if (headerBytes.Length > 0)
        {
            stream.Write(headerBytes, 0, headerBytes.Length);
        }
        if (payload.Length > 0)
        {
            stream.Write(payload, 0, payload.Length);
        }
        stream.Flush();
    }

    private static IncomingPacket ReadPacket(NetworkStream stream)
    {
        byte[] header = new byte[PacketHeaderBytes];
        ReadExact(stream, header, header.Length);
        uint magic = ReadUInt32BE(header, 0);
        if (magic != PacketMagic)
        {
            throw new IOException("Invalid packet magic.");
        }
        byte version = header[4];
        if (version != PacketVersion)
        {
            throw new IOException("Unsupported packet version " + version);
        }
        byte type = header[5];
        uint headerLength = ReadUInt32BE(header, 8);
        ulong payloadLength = ReadUInt64BE(header, 12);
        if (headerLength > 1024 * 1024 || payloadLength > 256UL * 1024UL * 1024UL)
        {
            throw new IOException("Packet too large.");
        }

        byte[] headerBytes = new byte[(int)headerLength];
        if (headerBytes.Length > 0)
        {
            ReadExact(stream, headerBytes, headerBytes.Length);
        }
        byte[] payload = new byte[(int)payloadLength];
        if (payload.Length > 0)
        {
            ReadExact(stream, payload, payload.Length);
        }

        return new IncomingPacket
        {
            type = type,
            headerJson = headerBytes.Length == 0 ? "{}" : Utf8NoBom.GetString(headerBytes),
            payload = payload
        };
    }

    private static void ReadExact(Stream stream, byte[] buffer, int count)
    {
        int offset = 0;
        while (offset < count)
        {
            int read = stream.Read(buffer, offset, count - offset);
            if (read <= 0)
            {
                throw new EndOfStreamException();
            }
            offset += read;
        }
    }

    private struct IncomingPacket
    {
        public byte type;
        public string headerJson;
        public byte[] payload;
    }

    private static void WriteUInt16BE(byte[] buffer, int offset, ushort value)
    {
        buffer[offset] = (byte)(value >> 8);
        buffer[offset + 1] = (byte)value;
    }

    private static void WriteUInt32BE(byte[] buffer, int offset, uint value)
    {
        buffer[offset] = (byte)(value >> 24);
        buffer[offset + 1] = (byte)(value >> 16);
        buffer[offset + 2] = (byte)(value >> 8);
        buffer[offset + 3] = (byte)value;
    }

    private static void WriteUInt64BE(byte[] buffer, int offset, ulong value)
    {
        for (int i = 7; i >= 0; i--)
        {
            buffer[offset + (7 - i)] = (byte)(value >> (i * 8));
        }
    }

    private static uint ReadUInt32BE(byte[] buffer, int offset)
    {
        return ((uint)buffer[offset] << 24) |
               ((uint)buffer[offset + 1] << 16) |
               ((uint)buffer[offset + 2] << 8) |
               buffer[offset + 3];
    }

    private static ulong ReadUInt64BE(byte[] buffer, int offset)
    {
        ulong value = 0;
        for (int i = 0; i < 8; i++)
        {
            value = (value << 8) | buffer[offset + i];
        }
        return value;
    }

    private string BuildMetadataRow(FrameRecord record)
    {
        StringBuilder row = new StringBuilder(1024);
        Append(row, record.frameIndex);
        Append(row, record.appTime);
        Append(row, record.frameTimestampNs);
        Append(row, record.refTimestampUs);
        Append(row, record.acquireStartUnixUs);
        Append(row, record.acquireEndUnixUs);
        Append(row, record.acquireMidUnixUs);
        Append(row, record.captureResult);
        Append(row, record.frameStatus);
        Append(row, ShouldSendHevcPayload() ? "video.h265" : string.Empty);
        Append(row, ShouldEncodeFrames() ? (ShouldSendHevcPayload() ? "hevc" : "hevc_discarded") : "none");
        Append(row, record.width);
        Append(row, record.height);
        Append(row, record.encoderWidth);
        Append(row, record.encoderHeight);
        Append(row, record.acquireMs);
        Append(row, record.copyMs);
        Append(row, record.downscaleMs);
        Append(row, record.encodeMs);
        Append(row, record.encoderDequeueInputMs);
        Append(row, record.encoderColorConvertMs);
        Append(row, record.encoderInputPutMs);
        Append(row, record.encoderQueueInputMs);
        Append(row, record.encoderDrainOutputMs);
        Append(row, record.encoderJavaTotalMs);
        Append(row, record.directBufferCreateMs);
        Append(row, record.encodeCallWallMs);
        Append(row, record.codecInputCopyOrConvertMs);
        Append(row, record.jniBridgeMs);
        Append(row, record.networkSendQueueMs);
        Append(row, record.encoderInputPath);
        Append(row, record.gazeMs);
        Append(row, record.frameIntervalMs);
        Append(row, record.gazeSampleAppTime);
        Append(row, record.xrHeadSampleAppTime);
        Append(row, record.xrHeadSampleUnixUs);
        Append(row, record.gazeSampleUnixUs);
        Append(row, record.gaze.valid);
        Append(row, record.gaze.source);
        Append(row, record.gaze.failureReason);
        Append(row, record.gaze.motionTrackingDataResult);
        Append(row, record.gaze.combinedPoseValid);
        Append(row, record.gaze.combinedStatus);
        Append(row, record.gaze.eyePosePositionRaw.x);
        Append(row, record.gaze.eyePosePositionRaw.y);
        Append(row, record.gaze.eyePosePositionRaw.z);
        Append(row, record.gaze.eyePosePositionUnity.x);
        Append(row, record.gaze.eyePosePositionUnity.y);
        Append(row, record.gaze.eyePosePositionUnity.z);
        Append(row, record.gaze.eyePoseRotationRaw.x);
        Append(row, record.gaze.eyePoseRotationRaw.y);
        Append(row, record.gaze.eyePoseRotationRaw.z);
        Append(row, record.gaze.eyePoseRotationRaw.w);
        Append(row, record.gaze.eyePoseRotationUnity.x);
        Append(row, record.gaze.eyePoseRotationUnity.y);
        Append(row, record.gaze.eyePoseRotationUnity.z);
        Append(row, record.gaze.eyePoseRotationUnity.w);
        Append(row, record.gaze.gazeWorldDirection.x);
        Append(row, record.gaze.gazeWorldDirection.y);
        Append(row, record.gaze.gazeWorldDirection.z);
        Append(row, record.gaze.legacyEyeDirection.x);
        Append(row, record.gaze.legacyEyeDirection.y);
        Append(row, record.gaze.legacyEyeDirection.z);
        Append(row, record.xrHeadPoseValid);
        Append(row, record.xrHeadPoseSource);
        Append(row, record.xrHeadPoseFailureReason);
        Append(row, record.xrHeadPose.position.x);
        Append(row, record.xrHeadPose.position.y);
        Append(row, record.xrHeadPose.position.z);
        Append(row, record.xrHeadPose.rotation.x);
        Append(row, record.xrHeadPose.rotation.y);
        Append(row, record.xrHeadPose.rotation.z);
        Append(row, record.xrHeadPose.rotation.w);
        Append(row, record.headPose.position.x);
        Append(row, record.headPose.position.y);
        Append(row, record.headPose.position.z);
        Append(row, record.headPose.rotation.x);
        Append(row, record.headPose.rotation.y);
        Append(row, record.headPose.rotation.z);
        Append(row, record.headPose.rotation.w);
        Append(row, record.rgbPose.position.x);
        Append(row, record.rgbPose.position.y);
        Append(row, record.rgbPose.position.z);
        Append(row, record.rgbPose.rotation.x);
        Append(row, record.rgbPose.rotation.y);
        Append(row, record.rgbPose.rotation.z);
        Append(row, record.rgbPose.rotation.w);
        Append(row, effectiveIntrinsics.fx);
        Append(row, effectiveIntrinsics.fy);
        Append(row, effectiveIntrinsics.cx);
        Append(row, effectiveIntrinsics.cy);
        Append(row, record.diagnosticPipelineMode);
        Append(row, record.captureScheduleMode);
        Append(row, record.unityFrameCount);
        Append(row, record.unityFramesSincePreviousCapture);
        Append(row, record.unityUnscaledDeltaMs);
        Append(row, record.schedulerLatenessMs);
        Append(row, record.schedulerSkippedSlots);
        Append(row, record.cameraTimestampDeltaNs);
        Append(row, record.cameraFrameIsNew);
        Append(row, record.cameraTimestampNonMonotonic);
        Append(row, record.captureProcessingMs);
        Append(row, record.streamInitializationMs);
        Append(row, record.encoderSamplesThisFrame);
        Append(row, record.encoderBytesThisFrame);
        Append(row, record.sendQueueDepthBefore);
        Append(row, record.sendQueueDepthAfterProcessing);
        Append(row, record.encoderPresentationTimeUs);
        Append(row, record.cameraPollIndex);
        Append(row, record.cameraPollsSincePreviousOutput);
        Append(row, record.observedNewCameraFramesSincePreviousOutput);
        Append(row, record.rateLimitedNewCameraFramesSincePreviousOutput);
        Append(row, record.pollIntervalMs);
        AppendLast(row, record.observedCameraTimestampDeltaNs);
        return row.ToString();
    }

    private string BuildTimestampRow(FrameRecord record)
    {
        StringBuilder row = new StringBuilder(256);
        Append(row, record.frameIndex);
        Append(row, record.refTimestampUs);
        Append(row, record.acquireMidUnixUs);
        Append(row, record.acquireStartUnixUs);
        Append(row, record.acquireEndUnixUs);
        Append(row, record.frameTimestampNs);
        Append(row, record.xrHeadSampleUnixUs);
        AppendLast(row, record.gazeSampleUnixUs);
        return row.ToString();
    }

    private static string MetadataHeader()
    {
        return "frame_index,app_time_seconds,frame_timestamp_ns,ref_timestamp_us,acquire_start_unix_us,acquire_end_unix_us,acquire_mid_unix_us," +
               "capture_result,frame_status,video_file,encoded_format,width,height,encoder_width,encoder_height," +
               "acquire_ms,copy_ms,downscale_ms,encode_ms,encoder_dequeue_input_ms,encoder_color_convert_ms,encoder_input_put_ms,encoder_queue_input_ms,encoder_drain_output_ms,encoder_java_total_ms," +
               "direct_buffer_create_ms,encode_call_wall_ms,codec_input_copy_or_convert_ms,jni_bridge_ms,network_send_queue_ms,encoder_input_path,gaze_ms,frame_interval_ms," +
               "gaze_sample_app_time_seconds,xr_head_sample_app_time_seconds,xr_head_sample_unix_us,gaze_sample_unix_us," +
               "gaze_valid,gaze_source,gaze_failure_reason,motion_tracking_data_result,combined_pose_valid,gaze_status," +
               "eye_pose_position_raw_x,eye_pose_position_raw_y,eye_pose_position_raw_z," +
               "eye_pose_position_unity_x,eye_pose_position_unity_y,eye_pose_position_unity_z," +
               "eye_pose_rotation_raw_x,eye_pose_rotation_raw_y,eye_pose_rotation_raw_z,eye_pose_rotation_raw_w," +
               "eye_pose_rotation_unity_x,eye_pose_rotation_unity_y,eye_pose_rotation_unity_z,eye_pose_rotation_unity_w," +
               "gaze_world_direction_x,gaze_world_direction_y,gaze_world_direction_z," +
               "legacy_eye_direction_x,legacy_eye_direction_y,legacy_eye_direction_z," +
               "xr_head_valid,xr_head_source,xr_head_failure_reason,xr_head_pos_x,xr_head_pos_y,xr_head_pos_z,xr_head_rot_x,xr_head_rot_y,xr_head_rot_z,xr_head_rot_w," +
               "head_pos_x,head_pos_y,head_pos_z,head_rot_x,head_rot_y,head_rot_z,head_rot_w," +
               "rgb_pos_x,rgb_pos_y,rgb_pos_z,rgb_rot_x,rgb_rot_y,rgb_rot_z,rgb_rot_w," +
               "fx,fy,cx,cy," +
               "diagnostic_pipeline_mode,capture_schedule_mode,unity_frame_count,unity_frames_since_previous_capture,unity_unscaled_delta_ms," +
               "scheduler_lateness_ms,scheduler_skipped_slots,camera_timestamp_delta_ns,camera_frame_is_new,camera_timestamp_non_monotonic," +
               "capture_processing_ms,stream_initialization_ms,encoder_samples_this_frame,encoder_bytes_this_frame,send_queue_depth_before,send_queue_depth_after_processing," +
               "encoder_presentation_time_us,camera_poll_index,camera_polls_since_previous_output,observed_new_camera_frames_since_previous_output," +
               "rate_limited_new_camera_frames_since_previous_output,poll_interval_ms,observed_camera_timestamp_delta_ns";
    }

    private static string TimestampHeader()
    {
        return "frame_index,ref_timestamp_us,pico_rgb_timestamp_us,pico_rgb_acquire_start_timestamp_us,pico_rgb_acquire_end_timestamp_us," +
               "pico_frame_timestamp_ns,pico_xr_head_timestamp_us,pico_gaze_timestamp_us";
    }

    private string BuildCameraJson(int sourceWidth, int sourceHeight, int encodedWidth, int encodedHeight, string encoderConfigJson)
    {
        StringBuilder json = new StringBuilder(1024);
        json.AppendLine("{");
        json.AppendLine("  \"image_width\": " + sourceWidth + ",");
        json.AppendLine("  \"image_height\": " + sourceHeight + ",");
        json.AppendLine("  \"encoded_width\": " + encodedWidth + ",");
        json.AppendLine("  \"encoded_height\": " + encodedHeight + ",");
        json.AppendLine("  \"target_fps\": " + targetFps + ",");
        json.AppendLine("  \"capture_schedule_mode\": " + JsonString(captureScheduleMode.ToString()) + ",");
        json.AppendLine("  \"diagnostic_pipeline_mode\": " + JsonString(diagnosticPipelineMode.ToString()) + ",");
        json.AppendLine("  \"diagnostic_update_status_every_frame\": " + JsonBool(diagnosticUpdateStatusEveryFrame) + ",");
        json.AppendLine("  \"display_refresh_hz\": " + JsonNumber(displayRefreshHz) + ",");
        json.AppendLine("  \"metadata_schema_version\": 2,");
        json.AppendLine("  \"frame_index_semantics\": " + JsonString("continuous_output_sequence") + ",");
        json.AppendLine("  \"capture_output_policy\": " + JsonString(UsesUniquePollingOutputPolicy() ? "unique_camera_timestamp_token_bucket" : "scheduled_attempt") + ",");
        json.AppendLine("  \"camera_poll_policy\": " + JsonString(UsesUniquePollingOutputPolicy() ? "once_per_unity_frame" : "scheduled") + ",");
        json.AppendLine("  \"duplicate_camera_frame_policy\": " + JsonString(UsesUniquePollingOutputPolicy() ? "drop_before_tracking_encoding_and_transport" : "emit_with_diagnostic_flag") + ",");
        json.AppendLine("  \"output_rate_clock\": " + JsonString(UsesUniquePollingOutputPolicy() ? "pico_camera_timestamp_ns" : "unity_realtime") + ",");
        json.AppendLine("  \"unique_polling_max_output_credit_frames\": " + JsonNumber(UniquePollingMaxOutputCredits) + ",");
        json.AppendLine("  \"unique_polling_warmup_timeout_seconds\": " + JsonNumber(uniquePollingWarmupTimeoutSeconds) + ",");
        json.AppendLine("  \"unique_polling_max_consecutive_acquire_failures\": " + uniquePollingMaxConsecutiveAcquireFailures + ",");
        json.AppendLine("  \"unique_polling_max_consecutive_timestamp_regressions\": " + uniquePollingMaxConsecutiveTimestampRegressions + ",");
        json.AppendLine("  \"unique_polling_no_new_frame_timeout_seconds\": " + JsonNumber(uniquePollingNoNewFrameTimeoutSeconds) + ",");
        json.AppendLine("  \"hevc_use_source_resolution\": " + JsonBool(hevcUseSourceResolution) + ",");
        json.AppendLine("  \"source_pixel_format\": " + JsonString(PixelFormat) + ",");
        json.AppendLine("  \"encoder_input_pixel_format\": " + JsonString(ShouldEncodeFrames() ? PixelFormat : "none") + ",");
        json.AppendLine("  \"encoder_input_mode\": " + JsonString(ShouldEncodeFrames() ? hevcInputMode.ToString() : "none") + ",");
        json.AppendLine("  \"encoder_input_downscale\": " + JsonString(ShouldEncodeFrames() && (sourceWidth != encodedWidth || sourceHeight != encodedHeight) ? "nearest_nv21" : "none") + ",");
        json.AppendLine("  \"stream_video_file\": " + JsonString(ShouldSendHevcPayload() ? "video.h265" : string.Empty) + ",");
        json.AppendLine("  \"stream_video_format\": " + JsonString(ShouldSendHevcPayload() ? "hevc_elementary_stream" : "none") + ",");
        json.AppendLine("  \"timestamp_standard\": " + JsonString("unix_epoch_microseconds_utc") + ",");
        json.AppendLine("  \"ref_timestamp_us_source\": " + JsonString("acquire_mid_unix_us") + ",");
        json.AppendLine("  \"raw_intrinsics_from_sdk\": {");
        json.AppendLine("    \"fx\": " + JsonNumber(cameraParameters.fx) + ",");
        json.AppendLine("    \"fy\": " + JsonNumber(cameraParameters.fy) + ",");
        json.AppendLine("    \"cx\": " + JsonNumber(cameraParameters.cx) + ",");
        json.AppendLine("    \"cy\": " + JsonNumber(cameraParameters.cy));
        json.AppendLine("  },");
        json.AppendLine("  \"extrinsics_head_to_rgb_camera\": {");
        json.AppendLine("    \"x\": " + JsonNumber(cameraParameters.x) + ",");
        json.AppendLine("    \"y\": " + JsonNumber(cameraParameters.y) + ",");
        json.AppendLine("    \"z\": " + JsonNumber(cameraParameters.z) + ",");
        json.AppendLine("    \"rx\": " + JsonNumber(cameraParameters.rx) + ",");
        json.AppendLine("    \"ry\": " + JsonNumber(cameraParameters.ry) + ",");
        json.AppendLine("    \"rz\": " + JsonNumber(cameraParameters.rz) + ",");
        json.AppendLine("    \"rw\": " + JsonNumber(cameraParameters.rw));
        json.AppendLine("  },");
        json.AppendLine("  \"encoder\": " + (string.IsNullOrEmpty(encoderConfigJson) ? "{}" : encoderConfigJson));
        json.AppendLine("}");
        return json.ToString();
    }

    private string BuildHelloJson()
    {
        return "{\"client\":\"pico_unity\",\"protocol_version\":1,\"package\":\"com.fudanfvl.picoegoproject\",\"target_fps\":" + targetFps +
               ",\"capture_schedule_mode\":" + JsonString(captureScheduleMode.ToString()) +
               ",\"diagnostic_pipeline_mode\":" + JsonString(diagnosticPipelineMode.ToString()) + "}";
    }

    private string BuildSessionStartingJson(string sessionName)
    {
        string outputPolicy = UsesUniquePollingOutputPolicy()
            ? "unique_camera_timestamp_token_bucket"
            : "scheduled_attempt";
        return "{\"event\":\"session_starting\",\"session_name\":" + JsonString(sessionName) +
               ",\"client\":\"pico_unity\",\"protocol_version\":1,\"metadata_schema_version\":2" +
               ",\"target_fps\":" + targetFps +
               ",\"frame_index_semantics\":\"continuous_output_sequence\"" +
               ",\"capture_output_policy\":" + JsonString(outputPolicy) +
               ",\"capture_schedule_mode\":" + JsonString(captureScheduleMode.ToString()) +
               ",\"diagnostic_pipeline_mode\":" + JsonString(diagnosticPipelineMode.ToString()) + "}";
    }

    private EffectiveIntrinsics CreateIntrinsics(RGBCameraParams raw)
    {
        return new EffectiveIntrinsics
        {
            fx = raw.fx,
            fy = raw.fy,
            cx = raw.cx,
            cy = raw.cy
        };
    }

    private static Pose ToUnityPose(Pose rightHandedPose)
    {
        return new Pose(ToUnityPosition(rightHandedPose.position), ToUnityRotation(rightHandedPose.rotation));
    }

    private static Vector3 ToUnityPosition(Vector3 rightHandedPosition)
    {
        return new Vector3(rightHandedPosition.x, rightHandedPosition.y, -rightHandedPosition.z);
    }

    private static Quaternion ToUnityRotation(Quaternion rightHandedRotation)
    {
        return new Quaternion(rightHandedRotation.x, rightHandedRotation.y, -rightHandedRotation.z, -rightHandedRotation.w);
    }

    private static Pose ToRgbCameraPose(RGBCameraParams parameters, Pose headPose)
    {
        Vector3 headToCameraPosition = new Vector3((float)parameters.x, (float)parameters.y, (float)parameters.z);
        Quaternion headToCameraRotation = new Quaternion((float)parameters.rx, (float)parameters.ry, (float)parameters.rz, (float)parameters.rw);
        Matrix4x4 headMatrix = Matrix4x4.TRS(headPose.position, headPose.rotation, Vector3.one);
        Matrix4x4 cameraMatrix = Matrix4x4.TRS(headToCameraPosition, headToCameraRotation, Vector3.one);
        Matrix4x4 rgbMatrix = headMatrix * cameraMatrix * Matrix4x4.Rotate(Quaternion.Euler(180f, 0f, 0f));
        Pose rightHandedRgbPose = new Pose(new Vector3(rgbMatrix.m03, rgbMatrix.m13, rgbMatrix.m23), rgbMatrix.rotation);
        return ToUnityPose(rightHandedRgbPose);
    }

    private static Vector3 ToVector3(PxrVector3f value)
    {
        return new Vector3(value.x, value.y, value.z);
    }

    private static Quaternion ToQuaternion(PxrVector4f value)
    {
        return new Quaternion(value.x, value.y, value.z, value.w);
    }

    private GazeSample CreateEmptyGaze(string failureReason)
    {
        return new GazeSample
        {
            source = "none",
            motionTrackingDataResult = int.MinValue,
            eyePoseRotationRaw = Quaternion.identity,
            eyePoseRotationUnity = Quaternion.identity,
            failureReason = failureReason
        };
    }

    private void PrepareStatusUi()
    {
        if (!showStatusUi)
        {
            return;
        }
        Canvas canvas = FindFirstObjectByType<Canvas>();
        if (canvas == null)
        {
            GameObject canvasObject = new GameObject("EgoStreamingStatusCanvas");
            canvas = canvasObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasObject.AddComponent<CanvasScaler>();
            canvasObject.AddComponent<GraphicRaycaster>();
        }

        GameObject textObject = new GameObject("EgoStreamingStatusText");
        textObject.transform.SetParent(canvas.transform, false);
        statusText = textObject.AddComponent<Text>();
        statusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        statusText.fontSize = 28;
        statusText.color = Color.white;
        statusText.alignment = TextAnchor.UpperLeft;
        RectTransform rect = statusText.rectTransform;
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = new Vector2(24f, -24f);
        rect.sizeDelta = new Vector2(-48f, 180f);
    }

    private void UpdateStatus(string message)
    {
        statusMessage = message;
        if (statusText != null)
        {
            statusText.text = message;
        }

        if (message == lastLoggedStatusMessage || Time.realtimeSinceStartup - lastStatusLogTime < statusLogIntervalSeconds)
        {
            return;
        }

        lastLoggedStatusMessage = message;
        lastStatusLogTime = Time.realtimeSinceStartup;
        Debug.Log("[EgoStreaming] " + message);
    }

    private void OnGUI()
    {
        if (!showStatusUi || statusText != null)
        {
            return;
        }

        if (guiStyle == null)
        {
            guiStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 32,
                alignment = TextAnchor.UpperLeft,
                normal = { textColor = Color.white }
            };
        }
        GUI.Label(new Rect(24, 24, Screen.width - 48, 180), statusMessage, guiStyle);
    }

    private void OnApplicationQuit()
    {
        Cleanup();
    }

    private void OnDestroy()
    {
        Cleanup();
    }

    private void Cleanup()
    {
        if (cleanedUp)
        {
            return;
        }

        cleanedUp = true;
        FinishEncoder();

        lock (sendLock)
        {
            networkStopRequested = true;
            Monitor.PulseAll(sendLock);
        }

        if (sendThread != null && sendThread.IsAlive && Thread.CurrentThread != sendThread)
        {
            sendThread.Join(1000);
        }

        try { networkStream?.Close(); } catch { }
        try { tcpClient?.Close(); } catch { }
        connected = false;

        if (sendThread != null && sendThread.IsAlive && Thread.CurrentThread != sendThread)
        {
            sendThread.Join(200);
        }
        if (receiveThread != null && receiveThread.IsAlive && Thread.CurrentThread != receiveThread)
        {
            receiveThread.Join(200);
        }

        if (cameraOpened)
        {
            try { PXR_Enterprise.CloseVSTCamera(); } catch { }
            cameraOpened = false;
        }
        TryStopEyeTracking();
        try { PXR_Enterprise.UnBindEnterpriseService(); } catch { }
    }

    private void TryStopEyeTracking()
    {
        if (eyeTrackingStartResult == int.MinValue)
        {
            return;
        }

        try
        {
            EyeTrackingStopInfo stopInfo = default;
            eyeTrackingStopResult = PXR_MotionTracking.StopEyeTracking(ref stopInfo);
        }
        catch
        {
            eyeTrackingStopResult = -1;
        }
    }

    private static float TicksToMilliseconds(long ticks)
    {
        return ticks * 1000f / System.Diagnostics.Stopwatch.Frequency;
    }

    private static long UnixTimeMicrosecondsNow()
    {
        return (DateTime.UtcNow.Ticks - UnixEpochTicks) / 10L;
    }

    private static string FormatEyeTrackingModes(EyeTrackingMode[] modes, int count)
    {
        if (modes == null || modes.Length == 0 || count <= 0)
        {
            return string.Empty;
        }

        int modeCount = Mathf.Min(count, modes.Length);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < modeCount; i++)
        {
            if (i > 0)
            {
                builder.Append('|');
            }
            builder.Append(modes[i]);
        }
        return builder.ToString();
    }

    private void Append(StringBuilder row, string value)
    {
        row.Append(EscapeCsv(value));
        row.Append(',');
    }

    private void Append(StringBuilder row, int value)
    {
        row.Append(value.ToString(invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, uint value)
    {
        row.Append(value.ToString(invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, long value)
    {
        row.Append(value.ToString(invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, ulong value)
    {
        row.Append(value.ToString(invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, float value)
    {
        row.Append(value.ToString("R", invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, double value)
    {
        row.Append(value.ToString("R", invariantCulture));
        row.Append(',');
    }

    private void Append(StringBuilder row, bool value)
    {
        row.Append(value ? "true" : "false");
        row.Append(',');
    }

    private void AppendLast(StringBuilder row, double value)
    {
        row.Append(value.ToString("R", invariantCulture));
    }

    private void AppendLast(StringBuilder row, long value)
    {
        row.Append(value.ToString(invariantCulture));
    }

    private string JsonNumber(double value)
    {
        return value.ToString("R", invariantCulture);
    }

    private string JsonNumber(float value)
    {
        return value.ToString("R", invariantCulture);
    }

    private static string JsonBool(bool value)
    {
        return value ? "true" : "false";
    }

    private static string JsonString(string value)
    {
        if (value == null)
        {
            return "null";
        }
        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r") + "\"";
    }

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }
        if (value.Contains(",") || value.Contains("\"") || value.Contains("\n") || value.Contains("\r"))
        {
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private static string ExtractJsonString(string json, string key, string fallback)
    {
        if (string.IsNullOrEmpty(json) || string.IsNullOrEmpty(key))
        {
            return fallback;
        }
        string needle = "\"" + key + "\"";
        int keyIndex = json.IndexOf(needle, StringComparison.Ordinal);
        if (keyIndex < 0)
        {
            return fallback;
        }
        int colon = json.IndexOf(':', keyIndex + needle.Length);
        if (colon < 0)
        {
            return fallback;
        }
        int firstQuote = json.IndexOf('"', colon + 1);
        if (firstQuote < 0)
        {
            return fallback;
        }
        StringBuilder value = new StringBuilder();
        bool escaped = false;
        for (int i = firstQuote + 1; i < json.Length; i++)
        {
            char ch = json[i];
            if (escaped)
            {
                value.Append(ch);
                escaped = false;
                continue;
            }
            if (ch == '\\')
            {
                escaped = true;
                continue;
            }
            if (ch == '"')
            {
                return value.ToString();
            }
            value.Append(ch);
        }
        return fallback;
    }

    private static long ExtractJsonLong(string json, string key, long fallback)
    {
        if (string.IsNullOrEmpty(json) || string.IsNullOrEmpty(key))
        {
            return fallback;
        }
        string needle = "\"" + key + "\"";
        int keyIndex = json.IndexOf(needle, StringComparison.Ordinal);
        if (keyIndex < 0)
        {
            return fallback;
        }
        int colon = json.IndexOf(':', keyIndex + needle.Length);
        if (colon < 0)
        {
            return fallback;
        }
        int start = colon + 1;
        while (start < json.Length && char.IsWhiteSpace(json[start]))
        {
            start++;
        }
        int end = start;
        if (end < json.Length && (json[end] == '-' || json[end] == '+'))
        {
            end++;
        }
        while (end < json.Length && char.IsDigit(json[end]))
        {
            end++;
        }
        if (end <= start)
        {
            return fallback;
        }
        if (long.TryParse(json.Substring(start, end - start), NumberStyles.Integer, CultureInfo.InvariantCulture, out long value))
        {
            return value;
        }
        return fallback;
    }

    private void QuitApplication()
    {
#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    private sealed class HevcEncoderWrapper : IDisposable
    {
        public string LastError { get; private set; } = string.Empty;
        public string ConfigJson { get; private set; } = "{}";
        public float LastJavaTotalMs { get; private set; }
        public float LastDequeueInputMs { get; private set; }
        public float LastColorConvertMs { get; private set; }
        public float LastInputPutMs { get; private set; }
        public float LastQueueInputMs { get; private set; }
        public float LastDrainOutputMs { get; private set; }
        public float LastDirectBufferCreateMs { get; private set; }
        public string LastInputPath { get; private set; } = string.Empty;
        public bool LastDirectFailureSafeForFallback { get; private set; }
        public long SubmittedInputFrameCount { get; private set; }
        public long MatchedOutputFrameCount { get; private set; }
        public long UnknownOutputPtsCount { get; private set; }
        public long DuplicateInputPtsCount { get; private set; }
        public long NonMonotonicInputPtsCount { get; private set; }
        public long PartialOutputSampleCount { get; private set; }
        public int FrameMetaHighWaterMark { get; private set; }
        public int FrameMetaCountAfterEos { get; private set; }
        public bool SawEndOfStream { get; private set; }

#if UNITY_ANDROID && !UNITY_EDITOR
        private AndroidJavaObject encoder;
#endif

        public int PendingSampleCount
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                return encoder == null ? 0 : encoder.Call<int>("getPendingSampleCount");
#else
                return 0;
#endif
            }
        }

        public bool Start(int width, int height, int bitRate, int frameRate, int iFrameIntervalSeconds)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                encoder = new AndroidJavaObject("com.fudanfvl.picoego.PicoHevcEncoder");
                bool ok = encoder.Call<bool>("start", width, height, bitRate, frameRate, iFrameIntervalSeconds);
                LastError = encoder.Call<string>("getLastError");
                ConfigJson = encoder.Call<string>("getConfigJson");
                ResetLifecycleStats();
                return ok;
            }
            catch (Exception ex)
            {
                LastError = ex.GetType().Name + ": " + ex.Message;
                return false;
            }
#else
            LastError = "HEVC encoder is only available on Android device builds.";
            return false;
#endif
        }

        public bool EncodeFrame(byte[] nv21, long presentationTimeUs, int frameIndex, long refTimestampUs)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                LastDirectBufferCreateMs = 0f;
                LastDirectFailureSafeForFallback = false;
                LastInputPath = "java_byte_array";
                int result = encoder.Call<int>("encodeFrame", nv21, presentationTimeUs, frameIndex, refTimestampUs);
                LastError = encoder.Call<string>("getLastError");
                RefreshLastTimingStats();
                return result >= 0;
            }
            catch (Exception ex)
            {
                LastError = ex.GetType().Name + ": " + ex.Message;
                LastInputPath = "java_byte_array_failed";
                return false;
            }
#else
            LastError = "HEVC encoder is only available on Android device builds.";
            return false;
#endif
        }

        public bool EncodeFrameDirect(IntPtr nv21Data, int byteCount, long presentationTimeUs, int frameIndex, long refTimestampUs)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            LastDirectFailureSafeForFallback = false;
            if (encoder == null)
            {
                LastError = "HEVC encoder is not started.";
                return false;
            }

            if (nv21Data == IntPtr.Zero || byteCount <= 0)
            {
                LastError = "Invalid direct NV21 frame pointer.";
                return false;
            }

            IntPtr directBuffer = IntPtr.Zero;
            bool javaEncodeInvoked = false;
            try
            {
                long startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
                unsafe
                {
                    directBuffer = AndroidJNI.NewDirectByteBuffer((byte*)nv21Data.ToPointer(), byteCount);
                }
                LastDirectBufferCreateMs = TicksToMilliseconds(System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
                if (directBuffer == IntPtr.Zero)
                {
                    LastError = "AndroidJNI.NewDirectByteBuffer returned null.";
                    LastDirectFailureSafeForFallback = true;
                    return false;
                }

                using (AndroidJavaObject directBufferObject = new AndroidJavaObject(directBuffer))
                {
                    LastInputPath = "direct_byte_buffer";
                    javaEncodeInvoked = true;
                    int result = encoder.Call<int>("encodeFrameDirect", directBufferObject, byteCount, presentationTimeUs, frameIndex, refTimestampUs);
                    LastError = encoder.Call<string>("getLastError");
                    RefreshLastTimingStats();
                    // Retry only failures that did not dequeue and retain a codec input buffer.
                    LastDirectFailureSafeForFallback = result == -2 || result == -3;
                    return result >= 0;
                }
            }
            catch (Exception ex)
            {
                LastError = ex.GetType().Name + ": " + ex.Message;
                LastInputPath = "direct_byte_buffer_failed";
                LastDirectFailureSafeForFallback = !javaEncodeInvoked;
                return false;
            }
            finally
            {
                if (directBuffer != IntPtr.Zero)
                {
                    AndroidJNI.DeleteLocalRef(directBuffer);
                }
            }
#else
            LastError = "HEVC encoder is only available on Android device builds.";
            return false;
#endif
        }

        private void RefreshLastTimingStats()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            LastJavaTotalMs = NsToMs(encoder.Call<long>("getLastTotalEncodeNs"));
            LastDequeueInputMs = NsToMs(encoder.Call<long>("getLastDequeueInputNs"));
            LastColorConvertMs = NsToMs(encoder.Call<long>("getLastColorConvertNs"));
            LastInputPutMs = NsToMs(encoder.Call<long>("getLastInputPutNs"));
            LastQueueInputMs = NsToMs(encoder.Call<long>("getLastQueueInputNs"));
            LastDrainOutputMs = NsToMs(encoder.Call<long>("getLastDrainOutputNs"));
#endif
        }

        private static float NsToMs(long value)
        {
            return value / 1000000f;
        }

        private void ResetLifecycleStats()
        {
            SubmittedInputFrameCount = 0L;
            MatchedOutputFrameCount = 0L;
            UnknownOutputPtsCount = 0L;
            DuplicateInputPtsCount = 0L;
            NonMonotonicInputPtsCount = 0L;
            PartialOutputSampleCount = 0L;
            FrameMetaHighWaterMark = 0;
            FrameMetaCountAfterEos = 0;
            SawEndOfStream = false;
        }

        private void RefreshLifecycleStats()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (encoder == null)
            {
                return;
            }

            SubmittedInputFrameCount = encoder.Call<long>("getSubmittedInputFrameCount");
            MatchedOutputFrameCount = encoder.Call<long>("getMatchedOutputFrameCount");
            UnknownOutputPtsCount = encoder.Call<long>("getUnknownOutputPtsCount");
            DuplicateInputPtsCount = encoder.Call<long>("getDuplicateInputPtsCount");
            NonMonotonicInputPtsCount = encoder.Call<long>("getNonMonotonicInputPtsCount");
            PartialOutputSampleCount = encoder.Call<long>("getPartialOutputSampleCount");
            FrameMetaHighWaterMark = encoder.Call<int>("getFrameMetaHighWaterMark");
            FrameMetaCountAfterEos = encoder.Call<int>("getFrameMetaCountAfterEos");
            SawEndOfStream = encoder.Call<bool>("getSawEndOfStream");
#endif
        }

        public void Finish(long presentationTimeUs)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (encoder != null)
            {
                try
                {
                    encoder.Call<int>("finish", presentationTimeUs);
                    LastError = encoder.Call<string>("getLastError");
                    RefreshLifecycleStats();
                }
                catch (Exception ex)
                {
                    LastError = ex.GetType().Name + ": " + ex.Message;
                }
            }
#endif
        }

        public string PeekSampleHeaderJson()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return encoder == null ? "{}" : encoder.Call<string>("peekSampleHeaderJson");
#else
            return "{}";
#endif
        }

        public byte[] PopSampleData()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return encoder == null ? Array.Empty<byte>() : encoder.Call<byte[]>("popSampleData");
#else
            return Array.Empty<byte>();
#endif
        }

        public void Dispose()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (encoder != null)
            {
                encoder.Call("stop");
                encoder.Dispose();
                encoder = null;
            }
#endif
        }
    }
}

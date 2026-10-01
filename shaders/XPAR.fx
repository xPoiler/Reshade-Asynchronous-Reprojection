// XPAR feed (part of FrameWarp / XPAR, GPL-3.0).
//
// For games without DLSS or FSR: hands ReShade's depth buffer to the XPAR add-on. XPAR estimates the
// motion between two game frames from the pictures themselves and works out the camera from that and
// the depth. The add-on enables this technique itself: there is nothing to enable or to set here. It
// needs ReShade's depth buffer set up for the game, as any depth-based effect does (Add-ons > Generic
// Depth, and the RESHADE_DEPTH_INPUT_* definitions).

#include "ReShade.fxh"

// Up to 1440p the depth is handed over as large as the picture; above, at half of it (the rest of XPAR
// works at the resolution a game would render at before upscaling).
#if BUFFER_HEIGHT > 1440
    #define XPAR_FEED_DIV 2
#else
    #define XPAR_FEED_DIV 1
#endif

// Set by the add-on when the depth comes the other way round (ReShade's RESHADE_DEPTH_INPUT_IS_REVERSED
// not matching the game): it checks that itself, so the setting does not have to be right.
uniform bool XPAR_Flip < hidden = true; > = false;

texture XPAR_Depth { Width = BUFFER_WIDTH / XPAR_FEED_DIV; Height = BUFFER_HEIGHT / XPAR_FEED_DIV; Format = R32F; };

float XPAR_FeedPS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    // ReShade.fxh orients the depth buffer (upside down, reversed, logarithmic, scaled) and then makes it
    // linear with a near plane of 1 and the configured far plane. XPAR wants the oriented buffer itself,
    // near = 1 and far = 0 (then it is proportional to 1 / distance): undo the last step and turn it round.
    const float far_plane = RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    const float lin = ReShade::GetLinearizedDepth(texcoord);
    const float depth = 1.0 - lin * far_plane / (1.0 + lin * (far_plane - 1.0));
    return XPAR_Flip ? 1.0 - depth : depth;
}

// A small probe of the depth and of the picture's brightness: the add-on reads it to tell which of the
// game's depth buffers belongs to the picture (it tries them in turn when the feed starts).
texture XPAR_Probe { Width = 256; Height = 144; Format = RG32F; };
sampler XPAR_DepthPoint { Texture = XPAR_Depth; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

float2 XPAR_ProbePS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    const float3 colour = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb;
    return float2(tex2Dlod(XPAR_DepthPoint, float4(texcoord, 0, 0)).x, sqrt(saturate(dot(colour, float3(0.299, 0.587, 0.114)))));
}

technique XPAR_Feed <
    hidden = true;
    ui_tooltip = "Depth for XPAR in games without DLSS or FSR. Enabled by the XPAR add-on itself.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = XPAR_FeedPS;
        RenderTarget = XPAR_Depth;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = XPAR_ProbePS;
        RenderTarget = XPAR_Probe;
    }
}

// XPAR feed (part of FrameWarp / XPAR, GPL-3.0).
//
// For games without DLSS or FSR: hands ReShade's depth buffer to the XPAR add-on. XPAR estimates the
// motion between two game frames from the pictures themselves and works out the camera from that and
// the depth. The add-on enables this technique itself: there is nothing to enable or to set here. It
// needs ReShade's depth buffer set up for the game, as any depth-based effect does (Add-ons > Generic
// Depth, and the RESHADE_DEPTH_INPUT_* definitions).

#include "ReShade.fxh"

// The depth is handed over at the size of the game's own depth buffer - the resolution it renders at before
// upscaling - set here by the add-on (XPAR_FEED_W/H), never larger than the picture: larger only costs every
// pass after it, with nothing in it the game's depth did not have. Until the add-on has set it: as large as
// the picture up to 1440p, half of it above.
#if BUFFER_HEIGHT > 1440
    #define XPAR_FEED_DIV 2
#else
    #define XPAR_FEED_DIV 1
#endif
#ifndef XPAR_FEED_W
    #define XPAR_FEED_W (BUFFER_WIDTH / XPAR_FEED_DIV)
#endif
#ifndef XPAR_FEED_H
    #define XPAR_FEED_H (BUFFER_HEIGHT / XPAR_FEED_DIV)
#endif

// Set by the add-on when the depth comes the other way round (ReShade's RESHADE_DEPTH_INPUT_IS_REVERSED
// not matching the game): it checks that itself, so the setting does not have to be right.
uniform bool XPAR_Flip < hidden = true; > = false;
// Set by the add-on when none of the game's depth buffers belongs to the picture: everything is handed over
// as far away, and XPAR warps the camera's turns only (they need no depth).
uniform bool XPAR_Far < hidden = true; > = false;

texture XPAR_Depth { Width = XPAR_FEED_W; Height = XPAR_FEED_H; Format = R32F; };

// A game that draws each frame into the next of two depth buffers: ReShade binds the one chosen, the add-on binds the other here. Which of them this
// frame drew into is seen in them: a few points of each are kept from the last frame (XPAR_Seen), and the second one is read when the chosen one did not
// change and it did. (Told apart from what the game binds, it went wrong: both are bound every frame, one to draw into, one to read last frame's from.)
uniform bool XPAR_Twin < hidden = true; > = false;
texture XPAR_TwinDepth : XPAR_TWIN_DEPTH;
sampler XPAR_TwinSampler { Texture = XPAR_TwinDepth; };
sampler XPAR_ChosenPoint { Texture = ReShade::DepthBufferTex; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };
sampler XPAR_TwinPoint { Texture = XPAR_TwinDepth; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };
texture XPAR_SeenChosen { Width = 16; Height = 9; Format = R32F; };
texture XPAR_SeenTwin { Width = 16; Height = 9; Format = R32F; };
texture XPAR_Pick { Width = 1; Height = 1; Format = R8; };
sampler XPAR_SeenChosenPoint { Texture = XPAR_SeenChosen; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };
sampler XPAR_SeenTwinPoint { Texture = XPAR_SeenTwin; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };
sampler XPAR_PickPoint { Texture = XPAR_Pick; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

float XPAR_PickPS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    if (!XPAR_Twin) return 0.0;
    float chosen = 0.0, twin = 0.0;
    for (int y = 0; y < 9; y++)
        for (int x = 0; x < 16; x++)
        {
            const float4 at = float4((x + 0.5) / 16.0, (y + 0.5) / 9.0, 0, 0);
            chosen += abs(tex2Dlod(XPAR_ChosenPoint, at).x - tex2Dlod(XPAR_SeenChosenPoint, at).x);
            twin += abs(tex2Dlod(XPAR_TwinPoint, at).x - tex2Dlod(XPAR_SeenTwinPoint, at).x);
        }
    return chosen == 0.0 && twin != 0.0 ? 1.0 : 0.0;
}
float XPAR_SeenChosenPS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target { return tex2Dlod(XPAR_ChosenPoint, float4(texcoord, 0, 0)).x; }
float XPAR_SeenTwinPS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target { return tex2Dlod(XPAR_TwinPoint, float4(texcoord, 0, 0)).x; }

float XPAR_GameDepth(float2 texcoord)
{
    // As ReShade.fxh orients the depth buffer (upside down, reversed, logarithmic, scaled), without its last step:
    // XPAR wants the oriented buffer itself, near = 1 and far = 0 (then it is proportional to 1 / distance).
#if RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN
    texcoord.y = 1.0 - texcoord.y;
#endif
#if RESHADE_DEPTH_INPUT_IS_MIRRORED
    texcoord.x = 1.0 - texcoord.x;
#endif
    texcoord.x /= RESHADE_DEPTH_INPUT_X_SCALE;
    texcoord.y /= RESHADE_DEPTH_INPUT_Y_SCALE;
#if RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET
    texcoord.x -= RESHADE_DEPTH_INPUT_X_PIXEL_OFFSET * BUFFER_RCP_WIDTH;
#else
    texcoord.x -= RESHADE_DEPTH_INPUT_X_OFFSET / 2.000000001;
#endif
#if RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET
    texcoord.y += RESHADE_DEPTH_INPUT_Y_PIXEL_OFFSET * BUFFER_RCP_HEIGHT;
#else
    texcoord.y += RESHADE_DEPTH_INPUT_Y_OFFSET / 2.000000001;
#endif
    const bool twin = tex2Dlod(XPAR_PickPoint, float4(0.5, 0.5, 0, 0)).x > 0.5;
    float depth = (twin ? tex2Dlod(XPAR_TwinSampler, float4(texcoord, 0, 0)).x : tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x) * RESHADE_DEPTH_MULTIPLIER;
#if RESHADE_DEPTH_INPUT_IS_LOGARITHMIC
    const float C = 0.01;
    depth = (exp(depth * log(C + 1.0)) - 1.0) / C;
#endif
#if !RESHADE_DEPTH_INPUT_IS_REVERSED
    depth = 1.0 - depth;
#endif
    return XPAR_Flip ? 1.0 - depth : depth;
}

float XPAR_FeedPS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    return XPAR_Far ? 0.0 : XPAR_GameDepth(texcoord);
}

// A small probe of the depth and of the picture's brightness: the add-on reads it to tell which of the
// game's depth buffers belongs to the picture (it tries them in turn when the feed starts). Always the
// game's depth, also while the far-away depth is handed over.
texture XPAR_Probe { Width = 256; Height = 144; Format = RG32F; };

float2 XPAR_ProbePS(in float4 position : SV_Position, in float2 texcoord : TEXCOORD) : SV_Target
{
    const float3 colour = tex2Dlod(ReShade::BackBuffer, float4(texcoord, 0, 0)).rgb;
    return float2(XPAR_GameDepth(texcoord), sqrt(saturate(dot(colour, float3(0.299, 0.587, 0.114)))));
}

technique XPAR_Feed <
    hidden = true;
    ui_tooltip = "Depth for XPAR in games without DLSS or FSR. Enabled by the XPAR add-on itself.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = XPAR_PickPS;
        RenderTarget = XPAR_Pick;
    }
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
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = XPAR_SeenChosenPS;
        RenderTarget = XPAR_SeenChosen;
    }
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = XPAR_SeenTwinPS;
        RenderTarget = XPAR_SeenTwin;
    }
}

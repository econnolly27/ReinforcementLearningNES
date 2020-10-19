-- RAM addresses: https://datacrystal.romhacking.net/wiki/Super_Mario_Bros.:RAM_map

local json = require("json")

local running = true;
local enable_box = true
local outstr;

local path_states = "C:\\sandbox\\mario-bm\\data\\maximumcoins\\states\\"
local path_imgs = "C:\\sandbox\\mario-bm\\data\\maximumcoins\\images\\"
local saveQ = true

-- draw a box and take care of coordinate checking
local function box(x1,y1,x2,y2,color)
	--gui.text(50,50,x1..","..y1.." "..x2..","..y2);
	if (disable_box) then
		if (x1 > 0 and x1 < 255 and x2 > 0 and x2 < 255 and y1 > 0 and y1 < 224 and y2 > 0 and y2 < 224) then
			gui.drawbox(x1,y1,x2,y2,color);
		end;
	end;
end;


-- hitbox coordinate offsets (x1,y1,x2,y2)
local mario_hb = 0x04AC; -- 1x4
local enemy_hb = 0x04B0; -- 5x4
local coin_hb  = 0x04E0; -- 3x4
local fiery_hb = 0x04C8; -- 2x4
local hammer_hb= 0x04D0; -- 9x4
local power_hb = 0x04C4; -- 1x4

-- addresses to check, to see whether the hitboxes should be drawn at all
local mario_ch = 0x000E;
local enemy_ch = 0x000F;
local coin_ch  = 0x0030;
local fiery_ch = 0x0024;
local hammer_ch= 0x002A;
local power_ch = 0x0014;
local mario_notscreen = 0x0033;

function main()
    emu.softreset()
    while (running) do
        count = emu.framecount()
        --emu.print(count)

        --start = os.clock()
        --while os.clock() - start < 0 do end

        --FCEU.frameadvance()

        --jp = joypad.read(1)
        --emu.print(jp)
        --joypad.write(1,{A=false, up=false, left=false, B=false, select=true, right=false, down=false, start=false})
		
		-- As YAML file
		outstr = "frame: "..count.." \n";
		state = {};
		state["frame"] = count
		-- from 0x04AC are about 0x48 addresse that indicate a hitbox
		-- different items use different addresses, some share
		-- there can for instance only be one powerup on screen at any time (the star in 1.1 gets replaced by the flower, if you get it)
		-- we cycle through the animation addresses for each type of hitbox, draw the corresponding hitbox if they are drawn
		-- we draw: mario (1), enemies (5), coins (3), hammers (9), powerups (1). (bowser and (his) fireball are considered enemies)

		-- mario
		-- String: <mario's state>, <x1,y1,x2,y2>
		if (memory.readbyte(mario_notscreen) > 0) then
			if (memory.readbyte(mario_hb) > 0) then 
				a,b,c,d = memory.readbyte(mario_hb),memory.readbyte(mario_hb+1),memory.readbyte(mario_hb+2),memory.readbyte(mario_hb+3);
				box(a,b,c,d, "green"); 
				outstr = outstr .. "mario: "..memory.readbyte(mario_ch)..","..a..","..b..","..c..","..d.."\n ";
				temp = {["lives"] = memory.readbyte(0x075A),
					["coins"] = memory.readbyte(0x075E),
					["world"] = memory.readbyte(0x075F),
					["level"] = memory.readbyte(0x0760),
					["state"] = memory.readbyte(mario_ch),
					["level_type"] = memory.readbyte(0x0773),
					["pos"] = {a,b,c,d}};
				state["mario"] = temp;
			end;
		end;
		
		-- enemies
		-- String: <enemy type>, <x1,y1,x2,y2>
		if (memory.readbyte(enemy_ch  ) > 0) then 
			a,b,c,d = memory.readbyte(enemy_hb),   memory.readbyte(enemy_hb+1), memory.readbyte(enemy_hb+2), memory.readbyte(enemy_hb+3);
			box(a,b,c,d, "green");
			outstr = outstr .. "Enemy 1: <"..memory.readbyte(0x0016).."> <"..a..","..b..","..c..","..d.."> \n";
			temp = {["type"] = memory.readbyte(0x0016), ["pos"] = {a,b,c,d}};
			state["enemy1"] = temp;
		end;
		if (memory.readbyte(enemy_ch+1) > 0) then 
			a,b,c,d = memory.readbyte(enemy_hb+4), memory.readbyte(enemy_hb+5), memory.readbyte(enemy_hb+6), memory.readbyte(enemy_hb+7);
			box(a,b,c,d, "green");
			outstr = outstr .. "Enemy 2: <"..memory.readbyte(0x0017).."> <"..a..","..b..","..c..","..d.."> \n";
			temp = {["type"] = memory.readbyte(0x0017), ["pos"] = {a,b,c,d}};
			state["enemy2"] = temp;
		end;
		if (memory.readbyte(enemy_ch+2) > 0) then 
			a,b,c,d = memory.readbyte(enemy_hb+8), memory.readbyte(enemy_hb+9), memory.readbyte(enemy_hb+10),memory.readbyte(enemy_hb+11);
			box(a,b,c,d, "green");
			outstr = outstr .. "Enemy 3: <"..memory.readbyte(0x0018).."> <"..a..","..b..","..c..","..d.."> \n";
			temp = {["type"] = memory.readbyte(0x0018), ["pos"] = {a,b,c,d}};
			state["enemy3"] = temp;
		end;
		if (memory.readbyte(enemy_ch+3) > 0) then 
			a,b,c,d = memory.readbyte(enemy_hb+12),memory.readbyte(enemy_hb+13),memory.readbyte(enemy_hb+14),memory.readbyte(enemy_hb+15);
			box(a,b,c,d, "green");
			outstr = outstr .. "Enemy 4: <"..memory.readbyte(0x0019).."> <"..a..","..b..","..c..","..d.."> \n";
			temp = {["type"] = memory.readbyte(0x0019), ["pos"] = {a,b,c,d}};
			state["enemy4"] = temp;
		end;
		if (memory.readbyte(enemy_ch+4) > 0) then 
			a,b,c,d = memory.readbyte(enemy_hb+16),memory.readbyte(enemy_hb+17),memory.readbyte(enemy_hb+18),memory.readbyte(enemy_hb+19)
			box(a,b,c,d, "green");
			outstr = outstr .. "Enemy 5: <"..memory.readbyte(0x001A).."> <"..a..","..b..","..c..","..d.."> \n";
			temp = {["type"] = memory.readbyte(0x001A), ["pos"] = {a,b,c,d}};
			state["enemy4"] = temp;
		end;

		-- coins
		if (memory.readbyte(coin_ch  ) > 0) then
			a,b,c,d = memory.readbyte(coin_hb),   memory.readbyte(coin_hb+1), memory.readbyte(coin_hb+2),  memory.readbyte(coin_hb+3);
			box(a,b,c,d, "green");
			temp = {["pos"] = {a,b,c,d}};
			state["coin1"] = temp;
		end;
		if (memory.readbyte(coin_ch+1) > 0) then
			a,b,c,d = memory.readbyte(coin_hb+4), memory.readbyte(coin_hb+5), memory.readbyte(coin_hb+6),  memory.readbyte(coin_hb+7);
			box(a,b,c,d, "green");
			temp = {["pos"] = {a,b,c,d}};
			state["coin2"] = temp;
		end;
		if (memory.readbyte(coin_ch+2) > 0) then
			a,b,c,d = memory.readbyte(coin_hb+8), memory.readbyte(coin_hb+9), memory.readbyte(coin_hb+10), memory.readbyte(coin_hb+11);
			box(a,b,c,d, "green");
			temp = {["pos"] = {a,b,c,d}};
			state["coin3"] = temp;
		end;
		
		-- (mario's) fireballs
		if (memory.readbyte(fiery_ch) > 0) then
			a,b,c,d = memory.readbyte(fiery_hb), memory.readbyte(fiery_hb+1), memory.readbyte(fiery_hb+2), memory.readbyte(fiery_hb+3);
			box(a,b,c,d, "green");
			temp = {["pos"] = {a,b,c,d}};
			state["fireball1"] = temp;
		end;
		if (memory.readbyte(fiery_ch+1) > 0) then
			a,b,c,d = memory.readbyte(fiery_hb+4), memory.readbyte(fiery_hb+5), memory.readbyte(fiery_hb+6), memory.readbyte(fiery_hb+7);
			box(a,b,c,d, "green");
			temp = {["pos"] = {a,b,c,d}};
			state["fireball2"] = temp;
		end;

		-- powerup
		if (memory.readbyte(power_ch) > 0) then
			a,b,c,d = memory.readbyte(power_hb),memory.readbyte(power_hb+1),memory.readbyte(power_hb+2),memory.readbyte(power_hb+3);
			box(a,b,c,d, "green");
			temp = {["state"] = memory.readbyte(0x0756), ["pos"] = {a,b,c,d}};
			state["powerup"] = temp;
		end;
		
		-- sounds
		if (memory.readbyte(0x00FD) > 0 and memory.readbyte(0x00FE) > 0 and memory.readbyte(0x00FF) > 0) then
			a,b,c = memory.readbyte(0x00FD),memory.readbyte(0x00FE),memory.readbyte(0x00FF);
			temp = {["state"] = {a,b,c}};
			state["sound"] = temp;
		end;
		
		jp = {joypad.read(1)}
		state['controller'] = jp
		state['sound'] = sound.get()
		
		if (saveQ and memory.readbyte(0x0770) > 0 and memory.readbyte(0x0033) > 0 and (memory.readbyte(mario_notscreen) > 0) and (memory.readbyte(mario_hb) > 0)
		and memory.readbyte(mario_ch) ~= 4 and memory.readbyte(mario_ch) ~= 5 and memory.readbyte(mario_ch) ~= 7) then
			local encode = json:encode(state);
			-- emu.print("------------------")
			-- emu.print(state)
			-- emu.print(encode);
			-- emu.print(sound.get());
			
			file = io.open(path_states..count.."_state.json", "w");
			file.write(file,encode);
			file.close(file);

			gui.savescreenshotas(path_imgs..count..".png");
			-- emu.print(count)
		end;
		
		-- 0x00FF -> Sound Effect Register 3 (jumping, bumping head, stomping, kicking, others) 
		-- gui.text(5,32, memory.readbyte(0x00FF));

		FCEU.frameadvance();
		state = {};
    end
end

main()


-- hammers
-- if (memory.readbyte(hammer_ch  ) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb), memory.readbyte(hammer_hb+1), memory.readbyte(hammer_hb+2), memory.readbyte(hammer_hb+3);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer1"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+1) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+4), memory.readbyte(hammer_hb+5), memory.readbyte(hammer_hb+6), memory.readbyte(hammer_hb+7);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer2"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+2) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+8), memory.readbyte(hammer_hb+9), memory.readbyte(hammer_hb+10),memory.readbyte(hammer_hb+11);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer3"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+3) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+12),memory.readbyte(hammer_hb+13),memory.readbyte(hammer_hb+14),memory.readbyte(hammer_hb+15);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer4"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+4) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+16),memory.readbyte(hammer_hb+17),memory.readbyte(hammer_hb+18),memory.readbyte(hammer_hb+19);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer5"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+5) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+20),memory.readbyte(hammer_hb+21),memory.readbyte(hammer_hb+22),memory.readbyte(hammer_hb+23);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer6"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+6) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+24),memory.readbyte(hammer_hb+25),memory.readbyte(hammer_hb+26),memory.readbyte(hammer_hb+27);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer7"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+7) > 0) then
--     a,b,c,d = memory.readbyte(hammer_hb+28),memory.readbyte(hammer_hb+29),memory.readbyte(hammer_hb+30),memory.readbyte(hammer_hb+31);
--     box(a,b,c,d, "green");
--     temp = {["pos"] = {a,b,c,d}};
--     state["hammer8"] = temp;
-- end;
-- if (memory.readbyte(hammer_ch+8) > 0) then
-- 	a,b,c,d = memory.readbyte(hammer_hb+32),memory.readbyte(hammer_hb+33),memory.readbyte(hammer_hb+34),memory.readbyte(hammer_hb+35);
-- 	box(a,b,c,d, "green");
-- 	temp = {["pos"] = {a,b,c,d}};
-- 	state["hammer9"] = temp;
-- end;
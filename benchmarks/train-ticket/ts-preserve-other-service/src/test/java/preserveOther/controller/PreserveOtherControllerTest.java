package preserveOther.controller;

import com.alibaba.fastjson.JSONObject;
import edu.fudan.common.util.Response;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.springframework.http.*;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import preserveOther.entity.OrderTicketsInfo;
import preserveOther.service.PreserveOtherService;

@RunWith(JUnit4.class)
public class PreserveOtherControllerTest {

    @InjectMocks
    private PreserveOtherController preserveOtherController;

    @Mock
    private PreserveOtherService preserveService;
    private MockMvc mockMvc;
    private Response response = new Response();

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        mockMvc = MockMvcBuilders.standaloneSetup(preserveOtherController).build();
    }

    @Test
    public void testHome() throws Exception {
        mockMvc.perform(MockMvcRequestBuilders.get("/api/v1/preserveotherservice/welcome"))
                .andExpect(MockMvcResultMatchers.status().isOk())
                .andExpect(MockMvcResultMatchers.content().string("Welcome to [ PreserveOther Service ] !"));
    }

    @Test
    public void testPreserve() throws Exception {
        OrderTicketsInfo oti = new OrderTicketsInfo();
        Mockito.when(preserveService.preserve(Mockito.any(OrderTicketsInfo.class), Mockito.any(HttpHeaders.class))).thenReturn(response);
        String requestJson = JSONObject.toJSONString(oti);
        String result = mockMvc.perform(MockMvcRequestBuilders.post("/api/v1/preserveotherservice/preserveOther").contentType(MediaType.APPLICATION_JSON).content(requestJson))
                .andExpect(MockMvcResultMatchers.status().isOk())
                .andReturn().getResponse().getContentAsString();
        Assert.assertEquals(response, JSONObject.parseObject(result, Response.class));
    }

}

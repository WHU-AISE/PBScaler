package ticketinfo.service;

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
import org.springframework.web.client.RestTemplate;
import ticketinfo.entity.Travel;

@RunWith(JUnit4.class)
public class TicketInfoServiceImplTest {

    @InjectMocks
    private TicketInfoServiceImpl ticketInfoServiceImpl;

    @Mock
    private RestTemplate restTemplate;

    private HttpHeaders headers = new HttpHeaders();

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testQueryForTravel() {
        Travel info = new Travel();
        HttpEntity requestEntity = new HttpEntity(info, headers);
        Response response = new Response();
        ResponseEntity<Response> re = new ResponseEntity<>(response, HttpStatus.OK);
        Mockito.when(restTemplate.exchange(
                "http://ts-basic-service:15680/api/v1/basicservice/basic/travel",
                HttpMethod.POST,
                requestEntity,
                Response.class)).thenReturn(re);
        Response result = ticketInfoServiceImpl.queryForTravel(info, headers);
        Assert.assertEquals(new Response<>(null, null, null), result);
    }

    @Test
    public void testQueryForStationId() {
        HttpEntity requestEntity = new HttpEntity(headers);
        Response response = new Response();
        ResponseEntity<Response> re = new ResponseEntity<>(response, HttpStatus.OK);
        Mockito.when( restTemplate.exchange(
                "http://ts-basic-service:15680/api/v1/basicservice/basic/name",
                HttpMethod.GET,
                requestEntity,
                Response.class)).thenReturn(re);
        Response result = ticketInfoServiceImpl.queryForStationId("name", headers);
        Assert.assertEquals(new Response<>(null, null, null), result);
    }

}
